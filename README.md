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

### 🐹 Run the code

The code below verifies the experimental results of the `mbpp` dataset.

```bash
python Experiments/run_mbpp.py
```

### Locally hosted LLMs (OpenAI-compatible servers)

Copy `template.local.env` to `.env` in the repository root. It contains the six
model names and LAN endpoints for your configuration. Alternatively export the
same variables in your shell; exported values take precedence over `.env`.

```bash
cp template.local.env .env
python Experiments/run_mbpp.py --batch_size 1 --epochs 1
# Run only the three small models (tier 0) or the three large models (tier 1):
python Experiments/run_mbpp.py --llm_tier 0 --batch_size 1 --epochs 1
python Experiments/run_mbpp.py --llm_tier 1 --batch_size 1 --epochs 1
```

All five experiment scripts support `--llm_tier`. Without it, the router selects
among all configured local models. The router embeds the detailed Name/Description profiles you added in
`MAR/LLM/llm_profile.py`, preserving their content and order. Endpoint configuration
only connects those model names to servers. Optional tier filtering restricts the
candidate pool; it does not change the learned routing policy. Prepare the
benchmark datasets and the existing experiment dependencies before running.
The model embedding encoder still uses `sentence-transformers/all-MiniLM-L6-v2`
(download or cache it beforehand for offline runs).

`SMALL_MODEL_NAMES`, `SMALL_ENDPOINTS`, and `SMALL_API_KEYS` are aligned JSON arrays
for tier 0. The corresponding `LARGE_*` arrays describe tier 1. Each name must be
unique and match the model ID served at the corresponding `/v1` base URL. To use
only one tier, set all three arrays of the other tier to `[]`. Invalid or partial
configuration fails before experiment setup. When local configuration is present,
unknown model names fail instead of making a hosted API request. With all six
variables unset, the authored profile list is used with the existing `URL`/`KEY`
client (for example, an OpenAI-compatible gateway exposing those model names).
For direct access to the six independent servers, configure `.env` as above.

Servers must implement `POST /v1/chat/completions` with OpenAI-compatible responses.
Empty API keys are supported using a non-secret SDK placeholder (`local-no-key`);
set a real key for servers requiring authentication. Both synchronous `gen` and
asynchronous `agen` route to each model's own endpoint, forward temperature,
completion count and token budget, and close their clients after use. The local
default output budget is 4096 tokens; callers can override `max_tokens`.

Reported server token usage updates the experiment counters. If usage is omitted,
no tokens are estimated. Local requests incur zero hosted API cost, so the existing
`--cost_rate` term does not penalize local compute; GPU/time costs are not measured.
Run experiments on a machine that can reach these private LAN addresses.

To test endpoint routing without GPUs, datasets, or your LAN servers:

```bash
python -m pip install openai python-dotenv class-registry tiktoken aiohttp requests groq tenacity "setuptools<81"
python -m unittest discover -s MAR/tests -v
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
