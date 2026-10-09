import asyncio
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import dotenv_values

from MAR.LLM import LLMRegistry, LocalChat
from MAR.LLM.local_config import load_local_models
from MAR.LLM.llm_profile import get_llm_profiles, hosted_llm_profile
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens


VARIABLES = [f'{group}_{suffix}' for group in ('SMALL', 'LARGE')
             for suffix in ('MODEL_NAMES', 'ENDPOINTS', 'API_KEYS')]
TEMPLATE = Path(__file__).resolve().parents[2] / 'template.local.env'


class ConfigFixture:
    def setUp(self):
        self.env = patch.dict(os.environ, {name: '' for name in VARIABLES})
        self.env.start()
        self.dotenv = patch('MAR.LLM.local_config.load_dotenv')
        self.dotenv.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.dotenv.stop)

    def configure(self):
        os.environ.update(dotenv_values(TEMPLATE))


class LocalConfigTests(ConfigFixture, unittest.TestCase):
    def test_hosted_mode_unchanged(self):
        self.assertEqual(load_local_models(), {})
        self.assertEqual(get_llm_profiles(), hosted_llm_profile)
        self.assertEqual(LLMRegistry.get().model_name, 'gpt-4o-mini')
        with self.assertRaisesRegex(ValueError, 'requires local'):
            get_llm_profiles(0)

    def test_all_six_model_endpoint_and_tier_mappings(self):
        self.configure()
        models = load_local_models()
        self.assertEqual(len(models), 6)
        for tier, group in enumerate(('SMALL', 'LARGE')):
            names = json.loads(os.environ[f'{group}_MODEL_NAMES'])
            endpoints = json.loads(os.environ[f'{group}_ENDPOINTS'])
            expected = [profile for profile in hosted_llm_profile if profile['Name'] in names]
            self.assertEqual(get_llm_profiles(tier), expected)
            for name, endpoint in zip(names, endpoints):
                model = models[name]
                self.assertEqual((model.endpoint, model.api_key, model.tier), (endpoint, '', tier))
                self.assertIsInstance(LLMRegistry.get(name), LocalChat)
        self.assertEqual(len(get_llm_profiles()), 6)
        self.assertEqual(get_llm_profiles(), hosted_llm_profile)
        self.assertEqual(LLMRegistry.get().model_name, next(iter(models)))
        with self.assertRaisesRegex(ValueError, 'not configured'):
            LLMRegistry.get('gpt-4o-mini')

    def test_missing_description_is_not_silently_replaced(self):
        self.configure()
        names = json.loads(os.environ['SMALL_MODEL_NAMES'])
        names[0] = 'new/model'
        os.environ['SMALL_MODEL_NAMES'] = json.dumps(names)
        with self.assertRaisesRegex(ValueError, 'Add Name/Description profiles'):
            get_llm_profiles()

    def test_invalid_configuration(self):
        for variable, value in (
            ('SMALL_MODEL_NAMES', 'not json'),
            ('SMALL_MODEL_NAMES', '{}'),
            ('SMALL_MODEL_NAMES', '[5]'),
            ('SMALL_MODEL_NAMES', '[""]'),
            ('SMALL_ENDPOINTS', '[]'),
            ('SMALL_ENDPOINTS', '["ftp://host", "http://host", "http://host"]'),
            ('SMALL_ENDPOINTS', '["http://user:secret@host", "http://host", "http://host"]'),
            ('SMALL_API_KEYS', '[null, "", ""]'),
        ):
            with self.subTest(variable=variable, value=value):
                self.configure()
                os.environ[variable] = value
                with self.assertRaises(ValueError):
                    load_local_models()
        self.configure()
        os.environ['LARGE_MODEL_NAMES'] = os.environ['SMALL_MODEL_NAMES']
        with self.assertRaisesRegex(ValueError, 'unique'):
            load_local_models()
        os.environ.update({name: '[]' for name in VARIABLES})
        with self.assertRaisesRegex(ValueError, 'at least one'):
            load_local_models()

    def test_one_tier_and_secret_not_in_repr(self):
        self.configure()
        for suffix in ('MODEL_NAMES', 'ENDPOINTS', 'API_KEYS'):
            os.environ[f'LARGE_{suffix}'] = '[]'
        os.environ['SMALL_API_KEYS'] = '["secret-value", "", ""]'
        self.assertNotIn('secret-value', repr(load_local_models()))
        self.assertEqual(len(get_llm_profiles()), 3)
        with self.assertRaisesRegex(ValueError, 'No local models'):
            get_llm_profiles(1)
        with self.assertRaisesRegex(ValueError, '0 or 1'):
            get_llm_profiles(2)


class LocalChatTests(ConfigFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.configure()
        self.requests = []
        captured = self.requests
        self.failures = [0]
        failures = self.failures

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((self.path, self.headers.get('Authorization'), payload))
                if failures[0]:
                    failures[0] -= 1
                    self.send_response(500)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                body = json.dumps({
                    'id': 'test', 'object': 'chat.completion', 'created': 0,
                    'model': payload['model'],
                    'choices': [{'index': i, 'message': {'role': 'assistant', 'content': f'answer {i}'},
                                 'finish_reason': 'stop'} for i in range(payload['n'])],
                    'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18},
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for group in ('SMALL', 'LARGE'):
            os.environ[f'{group}_ENDPOINTS'] = json.dumps([
                f'http://127.0.0.1:{self.server.server_port}/{group.lower()}/{i}/v1' for i in range(3)])
        for counter in (Cost, PromptTokens, CompletionTokens):
            counter.instance().reset()

    def test_sync_and_async_use_each_models_endpoint(self):
        for model in load_local_models().values():
            with self.subTest(model=model.name):
                client = LLMRegistry.get(model.name)
                self.assertEqual(client.gen('hello', max_tokens=123, temperature=0), 'answer 0')
                self.assertEqual(asyncio.run(client.agen('hello', max_tokens=123, temperature=0)), 'answer 0')
                for path, auth, payload in self.requests[-2:]:
                    self.assertEqual(path, model.endpoint.split(str(self.server.server_port))[1] + '/chat/completions')
                    self.assertEqual(auth, 'Bearer local-no-key')
                    self.assertEqual(payload, {'model': model.name, 'messages': [{'role': 'user', 'content': 'hello'}],
                                               'max_tokens': 123, 'temperature': 0, 'n': 1})
        self.assertEqual(PromptTokens.instance().value, 12 * 11)
        self.assertEqual(CompletionTokens.instance().value, 12 * 7)
        self.assertEqual(Cost.instance().value, 0)

    def test_keys_defaults_and_multiple_completions(self):
        os.environ['SMALL_API_KEYS'] = '["test-key", "", ""]'
        client = LLMRegistry.get()
        messages = [{'role': 'system', 'content': 'Be brief'}, {'role': 'user', 'content': 'Hello'}]
        self.assertEqual(client.gen(messages, num_comps=2), ['answer 0', 'answer 1'])
        self.assertEqual(asyncio.run(client.agen(messages, num_comps=2)), ['answer 0', 'answer 1'])
        for _, auth, payload in self.requests:
            self.assertEqual(auth, 'Bearer test-key')
            self.assertEqual(payload['messages'], messages)
            self.assertEqual(payload['max_tokens'], 4096)
            self.assertEqual(payload['temperature'], 1)

    def test_missing_content_and_usage(self):
        choice = SimpleNamespace(message=SimpleNamespace(content='ok'))
        self.assertEqual(LocalChat._response(SimpleNamespace(choices=[choice], usage=None)), 'ok')
        choice.message.content = None
        with self.assertRaisesRegex(ValueError, 'message content'):
            LocalChat._response(SimpleNamespace(choices=[choice], usage=None))
        with self.assertRaises(ValueError):
            LocalChat._response(SimpleNamespace(choices=[], usage=None))

    def test_observed_sdk_retries_usage_and_final_node_sync_async(self):
        from MAR.Experiment.trace import collect_traces, run_graph
        node = SimpleNamespace(id='worker', role=SimpleNamespace(role='Scientist'),
                               llm=LLMRegistry.get(), outputs=[])
        final = SimpleNamespace(id='final', llm=LLMRegistry.get(), outputs=[])
        def run(inputs, num_rounds):
            node.outputs.append(node.llm.gen(inputs['query']))
            final.outputs.append(asyncio.run(final.llm.agen(inputs['query'])))
            return final.outputs, 0
        graph = SimpleNamespace(nodes={'worker': node}, decision_node=final,
                                reasoning_name='Debate', run=run)
        self.failures[0] = 1
        with collect_traces() as traces:
            result = run_graph(graph, {'query': 'hello'}, 1)
        self.assertEqual(result[0], ['answer 0'])
        calls = traces[0]['calls']
        self.assertEqual(len(calls), 2)
        self.assertEqual([call['http_requests'] for call in calls], [2, 1])
        self.assertEqual(calls[0]['usage']['prompt_tokens'], 11)
        self.assertEqual(calls[1]['usage']['completion_tokens'], 7)
        self.assertTrue(calls[1]['is_final_node'])
        self.assertEqual(calls[0]['parameters']['max_tokens'], 4096)
        self.assertNotIn('gen', vars(node.llm))


if __name__ == '__main__':
    unittest.main()