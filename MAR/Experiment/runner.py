"""Shared test-only CLI and export setup for every benchmark runner."""
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import platform
import subprocess
from uuid import uuid4
from loguru import logger

from MAR.Experiment.logging import experiment_file_log
from MAR.Experiment.checkpoints import load_router_checkpoint
from MAR.Experiment.benchmarks import load_benchmark, score, sha256
from MAR.Experiment.evaluation import evaluate
from MAR.Experiment.results import ResultWriter
from MAR.Experiment.trace import utc_now


def add_evaluation_arguments(parser, benchmark):
    parser.add_argument('--mode', choices=['test', 'train-test'], default='test')
    parser.add_argument('--checkpoint', help='Trained router state_dict for evaluation or initialization')
    parser.add_argument('--allow_untrained', action='store_true')
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--max_train_samples', type=int, default=50,
                        help='Total training examples across epochs; default 50, 0 disables the limit')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--export_excel', action='store_true')
    parser.add_argument('--scoring_timeout', type=int, default=100)
    if benchmark in ('gsm8k', 'humaneval'):
        parser.add_argument('--train_dataset_json', default=None)
    if benchmark == 'math':
        from MAR.Experiment.benchmarks import ROOT
        parser.add_argument('--dataset_root', default=str(ROOT / 'Datasets/MATH'))


def validate_evaluation_arguments(parser, args, benchmark):
    if args.batch_size < 1 or args.epochs < 0 or args.max_agent < 1 or args.scoring_timeout < 1:
        parser.error('batch_size, max_agent and scoring_timeout must be positive; epochs must be nonnegative')
    if args.max_train_samples < 0:
        parser.error('max_train_samples must be nonnegative (0 means unlimited)')
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error('max_tasks must be positive')
    if not 0 <= args.start_epoch <= args.epochs:
        parser.error('start_epoch must be between zero and epochs')
    if args.mode == 'test':
        if args.start_epoch:
            parser.error('start_epoch is only for train-test')
        if not args.checkpoint and not args.allow_untrained:
            parser.error('Test evaluation requires --checkpoint or explicit --allow_untrained')
    elif benchmark == 'humaneval' and not args.train_dataset_json:
        parser.error('HumanEval has no official train split: provide disjoint --train_dataset_json or use --mode test')
    if args.mode == 'train-test' and not args.epochs and not args.checkpoint and not args.allow_untrained:
        parser.error('Zero epochs requires --checkpoint or --allow_untrained')
    if args.export_excel:
        import openpyxl


def run_test(benchmark, args, router=None, training_updates=0, evaluated_checkpoint=None, training_samples=0):
    with experiment_file_log(benchmark) as log_file:
        return _run_test(benchmark, args, router, training_updates,
                         evaluated_checkpoint, training_samples, log_file)


def _run_test(benchmark, args, router, training_updates, evaluated_checkpoint, training_samples, log_file):
    import torch
    from MAR.MasRouter.mas_router import MasRouter
    from MAR.LLM.llm_profile import get_llm_profiles
    from MAR.LLM.local_config import load_local_models
    from MAR.Agent.reasoning_profile import reasoning_profile
    from MAR.Prompts.tasks_profile import tasks_profile
    from MAR.Utils.utils import fix_random_seed
    fix_random_seed(args.seed)
    dataset = load_benchmark(benchmark, args)
    llms = get_llm_profiles(args.llm_tier)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_load = None
    supplied_router = router is not None
    if router is None:
        router = MasRouter(max_agent=args.max_agent, device=device).to(device)
        if args.checkpoint:
            checkpoint_load = load_router_checkpoint(router, args.checkpoint, device)
    if not training_updates and not evaluated_checkpoint and not args.checkpoint and not args.allow_untrained:
        raise ValueError('No trained router available; explicitly use --allow_untrained for a baseline')
    router.eval()
    checkpoint = evaluated_checkpoint or args.checkpoint
    weights_source = ("checkpoint_file" if checkpoint_load else "trained_in_memory" if training_updates
                      else "provided_router" if supplied_router else "random_initialization")
    logger.info("TEST START: benchmark={} weights_source={} checkpoint={} training_samples={} updates={}",
                benchmark, weights_source, checkpoint, training_samples, training_updates)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    metadata = {
        'benchmark': benchmark, 'split': 'test', 'protocol': dataset.protocol,
        'log_file': str(log_file),
        'inactive_test_arguments': ['lr', 'epochs', 'batch_size', 'cost_rate', 'num_rounds', 'decision_method', 'max_train_samples'],
        'started_at': utc_now(), 'arguments': vars(args), 'git_commit': commit, 'git_dirty': dirty,
        'python': platform.python_version(), 'packages': {d.metadata['Name']: d.version for d in distributions()},
        'device': str(device), 'seed': args.seed, 'dataset_size': len(dataset),
        'requested_tasks': min(args.max_tasks or len(dataset), len(dataset)),
        'dataset_sha256': dataset.sources, 'evaluation_order': 'Source file/record order',
        'checkpoint': checkpoint, 'checkpoint_sha256': checkpoint_load['sha256'] if checkpoint_load else sha256(checkpoint) if checkpoint else None,
        'checkpoint_load': checkpoint_load, 'evaluation_weights_source': weights_source,
        'router_initialization': 'checkpoint' if args.checkpoint else 'random', 'training_updates': training_updates, 'training_samples': training_samples,
        'training_sample_limit': args.max_train_samples,
        'training_grading': 'Same benchmark checker as test; invalid output scores zero; grader exceptions logged',
        'trained_on_separate_data_this_run': training_updates > 0, 'evaluation_batch_size': 1,
        'llm_profiles': llms, 'reasoning_profiles': reasoning_profile, 'prompt_sha256': sha256(args.prompt_file),
        'local_models': [{'name': m.name, 'endpoint': m.endpoint, 'tier': m.tier} for m in load_local_models().values()],
        'timing': 'task_seconds measures routing/embeddings and graph generation; scoring_seconds is separate',
        'llm_call_count': 'Logical generation calls including failures and final node; HTTP attempts include SDK retries',
        'cost_note': 'API cost only; local GPU/time costs are not measured',
        'metric': 'single-sample pass@1' if benchmark in ('mbpp', 'humaneval') else 'exact-match accuracy',
    }
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.result_file or f'results/{benchmark}/{now}_{uuid4().hex[:8]}/experiment.json'
    writer = ResultWriter(output, metadata)
    status = 'completed'
    try:
        with torch.inference_mode():
            evaluate(router, dataset, writer, tasks_profile, llms, reasoning_profile, args.prompt_file,
                     args.max_tasks, scorer=lambda response, row: score(benchmark, response, row, args.scoring_timeout),
                     task_label=0 if benchmark in ('gsm8k', 'math') else 2, benchmark=benchmark)
    except BaseException:
        status = 'interrupted'
        raise
    finally:
        metadata['finished_at'] = utc_now()
        writer.finish(status, excel=args.export_excel)
        logger.info('Results saved: {}', writer.path)
    logger.info("{} test accuracy: {}", benchmark, writer.report["summary"]["accuracy"])