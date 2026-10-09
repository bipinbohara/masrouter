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

### Held-out evaluation for all five benchmarks

All five runners default to `--mode test`, load only their designated evaluation
data, and use the same incremental JSON/JSONL/Excel research export. Test-only mode
never initializes an optimizer or partitions evaluation questions into training.
Every record is evaluated unless `--max_tasks` is supplied.

| Runner | Default evaluation source | Training behavior with `--mode train-test` |
| --- | --- | --- |
| `run_mmlu.py` | `Datasets/MMLU/data/test/*.csv` | Separate MMLU `dev` split |
| `run_gsm8k.py` | `Datasets/gsm8k/test.jsonl` | Separate `train.jsonl`, or `--train_dataset_json` |
| `run_math.py` | `Datasets/MATH/test/**/*.json` | Separate MATH `train/` directory |
| `run_mbpp.py` | Hugging Face MBPP `full/test` split | Separate MBPP `full/train` split |
| `run_humaneval.py` | Full `Datasets/humaneval/humaneval-py.jsonl` suite | No official training split; explicit disjoint `--train_dataset_json` required |

`--dataset_json` overrides the official test JSONL for GSM8K, HumanEval, or MBPP.
For GSM8K, supply a file containing the official test questions; the loader cannot
infer an unlabeled file's provenance. The legacy `gsm8k.jsonl` is not silently
repartitioned. For local MBPP JSONL, explicit split labels are respected; otherwise
only official test task IDs 11–510 are selected. An unlabelled mixed MBPP file is
therefore safe to use. `--dataset_root` overrides the MATH root (both train/test).
Training inputs overlapping GSM8K/HumanEval evaluation questions are rejected;
HumanEval task-ID overlaps are rejected too. MBPP import no longer downloads data
as a side effect; its default loader requires the parquet/Hugging Face dependencies
included in `requirements.txt`.

Install dependencies, configure the local endpoints in `.env`, and run each
benchmark with its corresponding trained router checkpoint:

```bash
python -m pip install -r requirements.txt
python Experiments/run_gsm8k.py --checkpoint gsm8k_router_epoch0_newnew.pth --export_excel
python Experiments/run_math.py --checkpoint math_router_epoch0_new.pth --export_excel
python Experiments/run_mbpp.py --checkpoint mbpp_router_epoch0_new.pth --export_excel
python Experiments/run_humaneval.py --checkpoint humaneval_router_epoch0.pth --export_excel
python Experiments/run_mmlu.py --checkpoint mmlu_router_epoch0.pth --export_excel
```

Use checkpoint paths matching the trained run you intend to compare. For a brief
setup check without weights, replace `--checkpoint ...` with
`--allow_untrained --max_tasks 10`. This is explicitly an untrained baseline.
`--llm_tier` and local endpoint routing work identically for every dataset.
Output defaults to a unique `results/<benchmark>/<run>/experiment.json` directory;
`--result_file results/<benchmark>/<custom-run>/experiment.json` sets an explicit
path. `--export_excel` creates the Excel workbook alongside the JSON artifacts.
`--batch_size` affects training only; test generation is measured one task at a time.

GSM8K evaluation compares signed numeric answers; MATH retains the project's
boxed-answer normalization/equivalence. MBPP executes its assertions. HumanEval
invokes `check(entry_point)` on the generated full function or prompt plus function
body. Code benchmarks record single-sample pass@1, not a multi-sample pass@k
estimate; MBPP retains the project's prompt format that exposes its assertion
list. Each coding task records the executed tests and grading details. The
`--scoring_timeout` option controls the existing code executor's timeout in seconds.
Generation timing (`task_seconds`) excludes grading; `scoring_seconds` and
`total_wall_seconds` allow separate reporting. Routing policies and training loss
remain unchanged; `train-test` retains the existing training objective.

### MMLU test evaluation and research exports

MMLU `dev` is used only by the optional training mode; reported evaluation uses
`Datasets/MMLU/data/test/*.csv`. The default mode now evaluates **every test
question**, including the first question and the last partial batch. It performs
no optimization on test data. Test questions are processed individually so task
latency includes the complete router/embedding and graph execution time.
`--batch_size` affects dev training only. Learned routing remains stochastic;
`--seed` records and initializes the random seed, while the dataset's existing
shuffle seed is 888.

Install the experiment dependencies with `python -m pip install -r requirements.txt`.
For evaluation with an existing trained router checkpoint:

```bash
python Experiments/run_mmlu.py --mode test \
  --checkpoint mmlu_router_epoch0.pth \
  --result_file results/mmlu/trained_test/experiment.json --export_excel
```

For a short connectivity/export check using an explicitly **untrained** router:

```bash
python Experiments/run_mmlu.py --mode test --allow_untrained --max_tasks 10 \
  --result_file results/mmlu/smoke_test/experiment.json --export_excel
```

Omit `--max_tasks` to evaluate the full test split. An untrained baseline must not
be reported as a trained MasRouter result. To retain the original dev training
objective before evaluation, use `--mode train-test --epochs 1 --batch_size 1`.
This retains the original training cap of 40 dev batches per epoch; test evaluation
no longer has the old 80-batch cap or skips batches based on the training count.
`--llm_tier` optionally restricts the local candidate pool as before.

Without `--result_file`, a unique directory under `results/mmlu/` is created.
Use a fresh output path for each run; existing result files are never overwritten.
Artifacts share the chosen filename prefix:

| Artifact | Contents |
| --- | --- |
| `experiment.json` | Run metadata, aggregate/per-subject results, and all test task records |
| `experiment.tasks.jsonl` | One completed/failed task per line, saved immediately |
| `experiment.summary.json` | Progress and summary, updated after each task |
| `experiment.xlsx` | Optional tasks, agents, calls, and run summary sheets |

Each task records its stable CSV file/row ID, subject, question, answer options,
reference answer, raw final response, parsed answer, correctness, selected agent
roles/models, final decision node, collaboration/rounds, raw node outputs, and
actual generation-call prompts/responses/errors. The selected integer agent count
is reported separately from the router's continuous agent-count prediction.
`task_seconds` includes routing and generation; `graph_seconds` measures graph
execution only. Per-call timing and server token usage are also recorded.

`total_llm_calls` counts logical generation invocations, including failed calls,
graph retries and the final decision node. Each local call's `http_requests` counts
SDK HTTP attempts, including internal SDK retries. Missing usage/request counts
are null and explicitly counted in the summary, rather than estimated as zero.
Errors that the original graph retries internally remain visible in the call list.
Failed tasks remain in the accuracy denominator. The parser accepts boxed or
explicit A/B/C/D answers and records ambiguous/unformatted answers as invalid.

Metadata includes the seed, package versions, Git commit/dirty state, checkpoint
and dataset hashes, prompt hash, and the complete authored model descriptions.
The summary provides micro accuracy, per-subject accuracy, timing, call counts,
known token usage, and model call distribution. Local API cost does not represent
GPU rental cost. Test records are captured by opt-in observation around graph
execution; the router's policy, role selection, collaboration and training loss
are unchanged. Training-mode progress/checkpoints retain the existing logs.

Normal completion and keyboard interruption consolidate the task journal into JSON;
a hard process kill still leaves completed JSONL records and the progress summary.
Excel is a convenience view: oversized cell text is marked as truncated, while
JSON retains full content. Large call tables are split across Excel sheets.
These artifacts include task and prompt content, but never API keys.

Validation without GPUs or live endpoints:

```bash
python -m unittest discover -s tests -v
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
