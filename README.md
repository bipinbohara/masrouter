# [ACL 2025] MasRouter: Learning to Route LLMs for Multi-Agent Systems

## 📰 News

- 🎉 Updates (2025-5-15) MasRouter is accpected to ACL 2025 Main!
- 🚩 Updates (2025-2-16) Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.11133).


## 🤔 Why MasRouter?

**MasRouter** expands LLM routing to the multi-agent systems (MAS) *for the first time*. It leverages the powerful reasoning capabilities of LLM MAS, while also making it relatively cost-effective.

![intro](assets/intro.png)

## 👋🏻 Method Overview

**MasRouter** integrates all components of MAS into a unified routing framework. It employs collaboration mode determination, role allocation, and LLM routing through a cascaded controller network, progressively constructing a MAS that balances effectiveness and efficiency.

![pipeline](assets/pipeline.png)

## 🏃‍♂️‍➡️ Quick Start

### 📊 Datasets

Please download the  `GSM8K`,  `HumanEval`, `MATH`, `MBPP`, `MMLU` datasets and place it in the `Datasets` folder. The file structure should be organized as follows:
```
Datasets
└── gsm8k
    └── gsm8k.jsonl
└── humaneval
    └── humaneval-py.jsonl
└── MATH
    └── test
    └── train
└── mbpp
    └── mbpp.jsonl
└── MMLU
    └── data
```

### 🔑 Add API keys

Add API keys in `template.env` and change its name to `.env`. We recommend that this API be able to access multiple LLMs.
```python
URL = "" # the URL of LLM backend
KEY = "" # the key for API
```

#### Multiple direct endpoints

MasRouter can also connect each model to a different OpenAI-compatible
endpoint without a gateway. Copy the included examples and fill in the API
keys (use `not-required` for servers that do not enforce authentication):

```bash
cp template.env .env
cp config.example.yaml config.yaml
```

Set `MASROUTER_MODEL_CONFIG="config.yaml"` in `.env`. Each `model_list` entry
defines the public model name, routing description, upstream model identifier,
endpoint, and API-key environment variable:

```yaml
model_list:
  - model_name: example/model
    tier: 0
    description: A fast model for straightforward tasks.
    litellm_params:
      model: openai/example/model
      api_base: http://model-server:80/v1
      api_key: os.environ/EXAMPLE_MODEL_API_KEY
```

The leading `openai/` in `litellm_params.model` selects the OpenAI-compatible
protocol and is removed before the model identifier is sent upstream. The
configured entries automatically replace the built-in model profile. If no
model configuration exists, MasRouter continues to use the shared `URL` and
`KEY` settings and its built-in model list.

### 🐹 Run the code

The code below verifies the experimental results of the `mbpp` dataset.

```bash
python experiments/run_mbpp.py
```

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:
```bibtex
@misc{yue2025masrouter,
      title={MasRouter: Learning to Route LLMs for Multi-Agent Systems}, 
      author={Yanwei Yue and Guibin Zhang and Boyang Liu and Guancheng Wan and Kun Wang and Dawei Cheng and Yiyan Qi},
      year={2025},
      eprint={2502.11133},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.11133}, 
}
```

## 🙏 Acknowledgement

Special thanks to the following repositories for their invaluable code and datasets:

- [MapCoder](https://github.com/Md-Ashraful-Pramanik/MapCoder)
- [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).
