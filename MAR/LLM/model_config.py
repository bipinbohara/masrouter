"""Load per-model OpenAI-compatible endpoints from a YAML configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


load_dotenv()


CONFIG_ENV_VAR = "MASROUTER_MODEL_CONFIG"
DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass(frozen=True)
class ModelRoute:
    """Connection details for one public model name."""

    model_name: str
    upstream_model: str
    api_base: str
    api_key: str
    description: str


def _config_path() -> Optional[Path]:
    configured_path = os.getenv(CONFIG_ENV_VAR)
    path = Path(configured_path or DEFAULT_CONFIG_PATH).expanduser()
    if configured_path and not path.is_file():
        raise FileNotFoundError(
            f"{CONFIG_ENV_VAR} points to a missing file: {path}"
        )
    return path if path.is_file() else None


def _required_string(value: Any, field: str, model_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Model {model_name!r} must define a non-empty {field!r} value"
        )
    return value.strip()


def _clean_api_base(value: Any, model_name: str) -> str:
    api_base = _required_string(value, "litellm_params.api_base", model_name)
    # Be forgiving when a URL was copied from rendered Markdown.
    if api_base.startswith("[") and "](" in api_base and api_base.endswith(")"):
        api_base = api_base.rsplit("](", 1)[1][:-1]
    return api_base.rstrip("/")


def _resolve_api_key(value: Any, model_name: str) -> str:
    if value is None:
        return "not-required"
    if not isinstance(value, str):
        raise ValueError(
            f"Model {model_name!r} has a non-string litellm_params.api_key"
        )

    value = value.strip()
    prefix = "os.environ/"
    if value.startswith(prefix):
        variable = value[len(prefix):]
        if not variable:
            raise ValueError(f"Model {model_name!r} has an empty environment variable")
        resolved = os.getenv(variable)
        if resolved is None:
            raise ValueError(
                f"Model {model_name!r} requires environment variable {variable!r}"
            )
        return resolved or "not-required"
    return value or "not-required"


def _upstream_model(value: Any, model_name: str) -> str:
    model = _required_string(value or model_name, "litellm_params.model", model_name)
    # Accept LiteLLM-style values. Here the leading ``openai/`` selects the
    # protocol; the remainder is the model identifier sent to the server.
    return model[len("openai/"):] if model.startswith("openai/") else model


def load_model_routes() -> Dict[str, ModelRoute]:
    """Return configured routes keyed by the public ``model_name`` alias.

    An absent default ``config.yaml`` means the legacy shared URL/KEY setup is
    in use. An explicitly configured missing or malformed file fails loudly.
    """

    path = _config_path()
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    model_list = config.get("model_list")
    if not isinstance(model_list, list) or not model_list:
        raise ValueError(f"{path} must contain a non-empty 'model_list'")

    routes: Dict[str, ModelRoute] = {}
    for index, entry in enumerate(model_list):
        if not isinstance(entry, dict):
            raise ValueError(f"model_list[{index}] must be a mapping")
        model_name = _required_string(entry.get("model_name"), "model_name", str(index))
        if model_name in routes:
            raise ValueError(f"Duplicate model_name in {path}: {model_name!r}")
        params = entry.get("litellm_params")
        if not isinstance(params, dict):
            raise ValueError(f"Model {model_name!r} must define 'litellm_params'")

        tier = entry.get("tier")
        default_description = f"Model {model_name}"
        if tier is not None:
            default_description += f" in tier {tier}"
        description = entry.get("description", default_description)
        description = _required_string(description, "description", model_name)

        routes[model_name] = ModelRoute(
            model_name=model_name,
            upstream_model=_upstream_model(params.get("model"), model_name),
            api_base=_clean_api_base(params.get("api_base"), model_name),
            api_key=_resolve_api_key(params.get("api_key"), model_name),
            description=description,
        )
    return routes


def configured_llm_profile() -> Optional[List[Dict[str, str]]]:
    """Build the MasRouter embedding profile from YAML, if configured."""

    routes = load_model_routes()
    if not routes:
        return None
    return [
        {"Name": route.model_name, "Description": route.description}
        for route in routes.values()
    ]


def route_for_model(model_name: str) -> Optional[ModelRoute]:
    """Look up connection details for a selected model."""

    return load_model_routes().get(model_name)
