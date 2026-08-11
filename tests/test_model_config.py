import importlib.util
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
MODULE_PATH = Path(__file__).parents[1] / "MAR" / "LLM" / "model_config.py"
SPEC = importlib.util.spec_from_file_location("masrouter_model_config", MODULE_PATH)
model_config = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model_config
SPEC.loader.exec_module(model_config)
configured_llm_profile = model_config.configured_llm_profile
load_model_routes = model_config.load_model_routes


class ModelConfigTest(unittest.TestCase):
    def _config(self, contents):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "models.yaml"
        path.write_text(textwrap.dedent(contents), encoding="utf-8")
        return str(path)

    def test_loads_litellm_style_route_and_profile(self):
        path = self._config(
            """
            model_list:
              - model_name: openai/gpt-oss-20b
                tier: 0
                description: Small reasoning model.
                litellm_params:
                  model: openai/openai/gpt-oss-20b
                  api_base: http://192.168.0.215:80/v1/
                  api_key: os.environ/GPT_OSS_20B_API_KEY
            """
        )
        environment = {
            "MASROUTER_MODEL_CONFIG": path,
            "GPT_OSS_20B_API_KEY": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            route = load_model_routes()["openai/gpt-oss-20b"]
            profile = configured_llm_profile()

        self.assertEqual(route.upstream_model, "openai/gpt-oss-20b")
        self.assertEqual(route.api_base, "http://192.168.0.215:80/v1")
        self.assertEqual(route.api_key, "secret")
        self.assertEqual(
            profile,
            [{"Name": "openai/gpt-oss-20b", "Description": "Small reasoning model."}],
        )

    def test_empty_key_uses_nonempty_placeholder(self):
        path = self._config(
            """
            model_list:
              - model_name: model-a
                litellm_params:
                  model: openai/model-a
                  api_base: http://localhost:8000/v1
                  api_key: os.environ/MODEL_A_KEY
            """
        )
        environment = {
            "MASROUTER_MODEL_CONFIG": path,
            "MODEL_A_KEY": "",
        }
        with patch.dict(os.environ, environment, clear=True):
            route = load_model_routes()["model-a"]
        self.assertEqual(route.api_key, "not-required")

    def test_explicit_missing_config_fails(self):
        with patch.dict(
            os.environ,
            {"MASROUTER_MODEL_CONFIG": "/missing/models.yaml"},
            clear=True,
        ):
            with self.assertRaises(FileNotFoundError):
                load_model_routes()

    def test_missing_referenced_key_fails(self):
        path = self._config(
            """
            model_list:
              - model_name: model-a
                litellm_params:
                  model: openai/model-a
                  api_base: http://localhost:8000/v1
                  api_key: os.environ/MODEL_A_KEY
            """
        )
        with patch.dict(
            os.environ,
            {"MASROUTER_MODEL_CONFIG": path},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "MODEL_A_KEY"):
                load_model_routes()

    def test_example_config_loads_all_six_routes(self):
        path = Path(__file__).parents[1] / "config.example.yaml"
        environment = {
            "MASROUTER_MODEL_CONFIG": str(path),
            "NEMOTRON_NANO_API_KEY": "unused",
            "GPT_OSS_20B_API_KEY": "unused",
            "QWEN_35B_API_KEY": "unused",
            "NEMOTRON_SUPER_API_KEY": "unused",
            "GPT_OSS_120B_API_KEY": "unused",
            "QWEN_122B_API_KEY": "unused",
        }
        with patch.dict(os.environ, environment, clear=True):
            routes = load_model_routes()

        self.assertEqual(len(routes), 6)
        self.assertEqual(
            routes["Qwen/Qwen3.5-122B-A10B-FP8"].api_base,
            "http://192.168.0.205:80/v1",
        )


if __name__ == "__main__":
    unittest.main()
