"""Configuration for independently hosted OpenAI-compatible chat models."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


@dataclass(frozen=True)
class LocalModel:
    name: str
    endpoint: str
    api_key: str = field(repr=False)
    tier: int


def load_local_models():
    """Read aligned JSON arrays; an absent configuration preserves hosted mode."""
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    groups = ('SMALL', 'LARGE')
    suffixes = ('MODEL_NAMES', 'ENDPOINTS', 'API_KEYS')
    if not any(os.getenv(f'{group}_{suffix}') for group in groups for suffix in suffixes):
        return {}
    models = {}
    for tier, group in enumerate(groups):
        arrays = []
        for suffix in suffixes:
            variable = f'{group}_{suffix}'
            try:
                values = json.loads(os.getenv(variable, '[]'))
            except json.JSONDecodeError:
                raise ValueError(f'{variable} must be a JSON array of strings') from None
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(f'{variable} must be a JSON array of strings')
            arrays.append(values)
        names, endpoints, keys = arrays
        if not (len(names) == len(endpoints) == len(keys)):
            raise ValueError(f'{group} model names, endpoints and API keys must have equal lengths')
        for name, endpoint, key in zip(names, endpoints, keys):
            if not name.strip() or name in models:
                raise ValueError('Local model names must be nonempty and unique across tiers')
            parsed = urlsplit(endpoint)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.query or parsed.fragment:
                raise ValueError(f'Endpoint for {name} must be an HTTP(S) base URL')
            if parsed.username is not None or parsed.password is not None:
                raise ValueError(f'Endpoint for {name} must not contain credentials; use API_KEYS')
            models[name] = LocalModel(name, endpoint.rstrip('/'), key, tier)
    if not models:
        raise ValueError('Local configuration must contain at least one model')
    return models
