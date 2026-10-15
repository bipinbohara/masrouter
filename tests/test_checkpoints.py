import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from MAR.Experiment.checkpoints import compatible_state_dict, load_router_checkpoint


def model(namespace):
    router = nn.Module()
    router.text_encoder = nn.Module()
    router.text_encoder.model = nn.Sequential(nn.Module())
    router.text_encoder.model[0].add_module(namespace, nn.Linear(3, 2))
    router.head = nn.Linear(2, 1)
    return router


class CheckpointTests(unittest.TestCase):
    def test_both_encoder_namespace_directions_load_every_tensor_and_metadata(self):
        for old, new in [('auto_model', 'model'), ('model', 'auto_model'), ('model', 'model')]:
            with self.subTest(old=old, new=new), tempfile.TemporaryDirectory() as directory:
                source, target = model(old), model(new)
                with torch.no_grad():
                    for i, tensor in enumerate(source.parameters()):
                        tensor.fill_(i + 1.25)
                path = Path(directory) / 'router.pth'
                torch.save(source.state_dict(), path)
                expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                info = load_router_checkpoint(target, path, 'cpu')
                adapted, _ = compatible_state_dict(source.state_dict(), target.state_dict())
                self.assertIn('text_encoder.model.0.' + new, adapted._metadata)
                for key, value in target.state_dict().items():
                    self.assertTrue(torch.equal(value, adapted[key]))
                self.assertEqual(info['sha256'], expected_hash)
                self.assertEqual(info['renamed_encoder_tensors'], 0 if old == new else 2)
                self.assertEqual(info['tensor_count'], 4)

    def test_incompatible_checkpoints_fail_before_mutating_model(self):
        for kind in ('missing', 'unexpected', 'shape', 'collision'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                target = model('model')
                before = {k: v.clone() for k,v in target.state_dict().items()}
                state = model('auto_model').state_dict()
                if kind == 'missing':
                    del state['head.bias']
                elif kind == 'unexpected':
                    state['unknown.weight'] = torch.zeros(1)
                elif kind == 'shape':
                    state['head.weight'] = torch.zeros(1, 5)
                else:
                    state['text_encoder.model.0.model.weight'] = state['text_encoder.model.0.auto_model.weight']
                path = Path(directory) / 'bad.pth'
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    load_router_checkpoint(target, path, 'cpu')
                for key, value in target.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))

    def test_original_checkpoint_bytes_are_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pth'
            torch.save(model('auto_model').state_dict(), path)
            before = path.read_bytes()
            load_router_checkpoint(model('model'), path, 'cpu')
            self.assertEqual(path.read_bytes(), before)

    def test_every_runner_and_resume_path_uses_shared_loader(self):
        for path in [*Path('Experiments').glob('run_*.py'), Path('MAR/Experiment/runner.py')]:
            source = path.read_text()
            self.assertNotIn('router.load_state_dict(torch.load(', source)
            self.assertIn('load_router_checkpoint(', source)
        self.assertIn('f"math_router_epoch{epoch}_new.pth"', Path('Experiments/run_math.py').read_text())

    def test_shared_test_runner_evaluates_loaded_weights_and_records_actual_hash(self):
        from MAR.Experiment.runner import run_test
        class Dataset:
            protocol = 'test fixture'
            sources = {}
            def __len__(self): return 1
        for benchmark in ('gsm8k', 'math', 'mbpp', 'humaneval'):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = model('auto_model')
                with torch.no_grad():
                    source.head.weight.fill_(42)
                checkpoint = root / 'router.pth'
                torch.save(source.state_dict(), checkpoint)
                prompt = root / 'prompt.json'
                prompt.write_text('{}')
                args = SimpleNamespace(seed=1234, max_agent=6, llm_tier=None, checkpoint=str(checkpoint),
                                       allow_untrained=False, prompt_file=str(prompt), max_tasks=1,
                                       result_file=str(root / 'experiment.json'), export_excel=False,
                                       max_train_samples=50)
                evaluated = []
                def evaluate(router, dataset, writer, *args, **kwargs):
                    self.assertTrue(torch.equal(router.head.weight, source.head.weight))
                    self.assertFalse(router.training)
                    self.assertFalse(torch.is_grad_enabled())
                    evaluated.append(True)
                with patch.dict(sys.modules, {
                    'MAR.MasRouter.mas_router': SimpleNamespace(MasRouter=lambda **kwargs: model('model')),
                    'MAR.Utils.utils': SimpleNamespace(fix_random_seed=lambda seed: None),
                    'MAR.Agent.reasoning_profile': SimpleNamespace(reasoning_profile=[]),
                }), patch('MAR.Experiment.runner.load_benchmark', return_value=Dataset()), \
                     patch('MAR.Experiment.runner.evaluate', side_effect=evaluate), \
                     patch('MAR.Experiment.logging.LOG_DIR', root / 'logs'):
                    run_test(benchmark, args)
                self.assertEqual(evaluated, [True])
                metadata = json.loads(Path(args.result_file).read_text())['metadata']
                self.assertEqual(metadata['evaluation_weights_source'], 'checkpoint_file')
                log = Path(metadata['log_file']).read_text()
                self.assertIn('Checkpoint LOADED', log)
                self.assertIn('TEST START', log)
                self.assertIn('test accuracy:', log)
                self.assertEqual(metadata['checkpoint_load']['renamed_encoder_tensors'], 2)
                self.assertEqual(metadata['checkpoint_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_actual_mmlu_initialization_loads_legacy_encoder_weights(self):
        tree = ast.parse(Path('Experiments/run_mmlu.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))
        load = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.checkpoint')
        with tempfile.TemporaryDirectory() as directory:
            source, target = model('auto_model'), model('model')
            path = Path(directory) / 'router.pth'
            torch.save(source.state_dict(), path)
            namespace = dict(args=SimpleNamespace(checkpoint=str(path)), router=target,
                             device='cpu', load_router_checkpoint=load_router_checkpoint)
            exec(compile(ast.Module(body=[load], type_ignores=[]), 'mmlu_init', 'exec'), namespace)
            self.assertTrue(torch.equal(target.head.weight, source.head.weight))
            self.assertEqual(namespace['checkpoint_load']['renamed_encoder_tensors'], 2)
