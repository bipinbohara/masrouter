"""Synchronous and asynchronous clients for local chat completion servers."""

from openai import OpenAI, AsyncOpenAI

from MAR.LLM.llm import LLM
from MAR.LLM.llm_registry import LLMRegistry
from MAR.LLM.local_config import load_local_models
from MAR.Utils.globals import PromptTokens, CompletionTokens
from MAR.Experiment.trace import instrument_client_kwargs, record_usage, record_parameters


@LLMRegistry.register('LocalChat')
class LocalChat(LLM):
    # Keep the completion budget practical for smaller local context windows.
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, model_name):
        self.model_name = model_name
        self.config = load_local_models()[model_name]

    def _client_kwargs(self):
        # The SDK requires a nonempty key even for servers without authentication.
        return {'base_url': self.config.endpoint,
                'api_key': self.config.api_key or 'local-no-key'}

    def _request(self, messages, max_tokens, temperature, num_comps):
        request = {'model': self.model_name,
                'messages': [{'role': 'user', 'content': messages}] if isinstance(messages, str) else messages,
                'max_tokens': self.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
                'temperature': self.DEFAULT_TEMPERATURE if temperature is None else temperature,
                'n': self.DEFUALT_NUM_COMPLETIONS if num_comps is None else num_comps}
        record_parameters(request)
        return request

    @staticmethod
    def _response(completion):
        record_usage(completion.usage)
        responses = [choice.message.content for choice in completion.choices]
        if not responses or any(response is None for response in responses):
            raise ValueError('Local server did not return chat message content')
        if completion.usage is not None:
            PromptTokens.instance().value += completion.usage.prompt_tokens or 0
            CompletionTokens.instance().value += completion.usage.completion_tokens or 0
        return responses[0] if len(responses) == 1 else responses

    def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        with OpenAI(**instrument_client_kwargs(self._client_kwargs())) as client:
            completion = client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, num_comps))
        return self._response(completion)

    async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        async with AsyncOpenAI(**instrument_client_kwargs(self._client_kwargs(), asynchronous=True)) as client:
            completion = await client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, num_comps))
        return self._response(completion)