from typing import Optional
from class_registry import ClassRegistry

from MAR.LLM.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        from MAR.LLM.local_config import load_local_models
        local_models = load_local_models()
        if model_name is None or model_name=="":
            model_name = next(iter(local_models), "gpt-4o-mini")
        if local_models:
            if model_name not in local_models:
                raise ValueError(f'Model {model_name} is not configured for local inference')
            return cls.registry.get('LocalChat', model_name)
        if 'DeepSeek-V3' in model_name:
            model = cls.registry.get('Deepseek', model_name)
        else:
            model = cls.registry.get('ALLChat', model_name)
        
        return model

