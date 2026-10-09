import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from importlib.metadata import distributions
from uuid import uuid4

from MAR.Experiment.results import ResultWriter
from MAR.Experiment.mmlu import evaluate
from MAR.Experiment.trace import utc_now
from MAR.LLM.local_config import load_local_models
import torch
import numpy as np
from loguru import logger
import torch.nn.functional as F

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import get_llm_profiles
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
from Datasets.mmlu_dataset import MMLUDataset
from Datasets.math_dataset import MATH_get_predict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(description="MAR Experiments on MMLU")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16,help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--domain', type=str, default="mmlu",help="Domain (the same as dataset name), default 'mmlu'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/mmlu.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=500.0)
    parser.add_argument('--max_agent', type=int, default=6)
    parser.add_argument("--llm_tier", type=int, choices=[0, 1], default=None,
                        help="Use only local tier 0 (small) or 1 (large); default uses both")
    parser.add_argument('--mode', choices=['test', 'train-test'], default='test',
                        help='Default evaluates test only; train-test retains dev training')
    parser.add_argument('--checkpoint', help='Router state_dict to load before evaluation/training')
    parser.add_argument('--allow_untrained', action='store_true',
                        help='Explicitly evaluate random router weights as an untrained baseline')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Limit test questions; omitted evaluates the entire test split')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--export_excel', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 0 or args.max_agent < 1:
        parser.error('batch_size/max_agent must be positive and epochs nonnegative')
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error('max_tasks must be positive')
    if args.mode == 'test' and not args.checkpoint and not args.allow_untrained:
        parser.error('Test-only evaluation requires --checkpoint or explicit --allow_untrained')
    if args.mode == 'test' and args.start_epoch:
        parser.error('--start_epoch applies only to --mode train-test')
    if args.mode == 'train-test' and not 0 <= args.start_epoch <= args.epochs:
        parser.error('--start_epoch must be between zero and --epochs')
    if args.mode == 'train-test' and args.epochs == 0 and not args.checkpoint and not args.allow_untrained:
        parser.error('Zero training epochs requires --checkpoint or --allow_untrained')
    if args.export_excel:
        import openpyxl  # Fail before sending requests if Excel support is missing.
    return args

def infinite_data_loader(dataset):
    perm = np.random.permutation(len(dataset))
    while True:
        for idx in perm:
            record = dataset[idx.item()]
            yield record


if __name__ == '__main__':
    args = parse_args()
    llms = get_llm_profiles(args.llm_tier)
    fix_random_seed(args.seed)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"mmlu_{current_time}.txt"
    configure_logging(log_name=log_file)
    total_solved, total_executed = (0, 0)

    dataset_train = MMLUDataset('dev') if args.mode == 'train-test' else None
    dataset_test = MMLUDataset('test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=args.max_agent, device=device).to(device)
    if args.checkpoint:
        router.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr) if args.mode == 'train-test' else None
    tasks = tasks_profile
    reasonings = reasoning_profile
    training_updates = 0
    if args.mode == 'train-test':
        logger.info("Start training...")

        train_batch = min(40,len(dataset_train)//args.batch_size)
        for i_epoch in range(args.epochs):
            if i_epoch < args.start_epoch:
                router.load_state_dict(torch.load(f"mmlu_router_epoch{i_epoch}.pth", map_location=device))
                continue
            for i_batch in range(train_batch):
                print(f"Batch {i_batch}",80*'-')
                start_ts = time.time()
                current_batch = dataloader(dataset_train, args.batch_size, i_batch)
                current_batch = [{"task":dataset_train.record_to_input(record)["task"], "answer":dataset_train.record_to_target_answer(record)} for row, record in current_batch.iterrows()]

                queries = [item['task'] for item in current_batch]
                answers = [item['answer'] for item in current_batch]
                task_labels = [1 for _ in current_batch]
                tasks_y = torch.tensor(task_labels).to(device)
                optimizer.zero_grad()
                results, costs, log_probs, tasks_probs, vae_loss, agents_num  = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)
                task_loss = F.cross_entropy(tasks_probs, tasks_y)
                utilities = []
                answers_loss = []
                is_solved_list = []
                for query, result, answer, log_prob, cost in zip(queries, results, answers, log_probs, costs):
                    predict_answer = MATH_get_predict(result)[0]
                    is_solved = str(predict_answer).strip()==str(answer).strip()
                    total_solved = total_solved + is_solved
                    total_executed = total_executed + 1
                    utility = is_solved - cost * args.cost_rate
                    utilities.append(utility)
                    is_solved_list.append(is_solved)
                    answer_loss = -log_prob * utility
                    answers_loss.append(answer_loss)
                    logger.debug(f"Raw Result: {result}")
                    logger.debug(f"Predict: {predict_answer}")
                    logger.debug(f"Truth: {answer}")
                    logger.debug(f"Cost: {cost}")
                    logger.debug(f"is_solved: {is_solved}")
                answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
                vae_loss = vae_loss.mean()
                is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [N, 1]
                # adjust_loss = ((1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num) + 0.25 * is_solved_tensor *  agents_num).mean()
                loss = task_loss + answer_loss + vae_loss*0.001 # + adjust_loss
                loss.backward()
                optimizer.step()
                training_updates += 1

                accuracy = total_solved / total_executed
                logger.info(f"Batch time {time.time() - start_ts:.3f}")
                logger.info(f"Accuracy: {accuracy}")

            logger.info(f"Epoch {i_epoch} Finishes",80*'-')
            torch.save(router.state_dict(), f"mmlu_router_epoch{i_epoch}.pth")

    router.eval()
    if args.mode == 'train-test' and not training_updates and not args.checkpoint and not args.start_epoch and not args.allow_untrained:
        raise ValueError('No training updates occurred; reduce batch_size or explicitly use --allow_untrained')
    result_file = args.result_file or f"results/mmlu/{current_time}_{uuid4().hex[:8]}/experiment.json"
    def digest(path):
        hasher = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    test_directory = Path(__file__).resolve().parents[1] / 'Datasets/MMLU/data/test'
    metadata = {
        'started_at': utc_now(), 'arguments': vars(args), 'git_commit': commit, 'git_dirty': dirty,
        'python': platform.python_version(), 'device': str(device),
        'packages': {dist.metadata['Name']: dist.version for dist in distributions()},
        'seed': args.seed, 'dataset_shuffle_seed': 888, 'split': 'test',
        'dataset_size': len(dataset_test),
        'requested_tasks': min(args.max_tasks or len(dataset_test), len(dataset_test)),
        'dataset_sha256': {p.name: digest(p) for p in sorted(test_directory.glob('*.csv'))},
        'llm_profiles': llms, 'reasoning_profiles': reasonings,
        'local_models': [{'name': model.name, 'endpoint': model.endpoint, 'tier': model.tier}
                         for model in load_local_models().values()],
        'prompt_sha256': digest(args.prompt_file),
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': digest(args.checkpoint) if args.checkpoint else None,
        'router_initialization': 'checkpoint' if args.checkpoint else 'random',
        'trained_on_dev_this_run': training_updates > 0, 'training_updates': training_updates,
        'resumed_checkpoint': f'mmlu_router_epoch{args.start_epoch-1}.pth' if args.start_epoch else None,
        'evaluated_checkpoint': f'mmlu_router_epoch{args.epochs-1}.pth' if training_updates or args.start_epoch else args.checkpoint,
        'evaluated_checkpoint_sha256': digest(f'mmlu_router_epoch{args.epochs-1}.pth') if training_updates or args.start_epoch else
                                      (digest(args.checkpoint) if args.checkpoint else None),
        'evaluation_batch_size': 1,
        'timing': 'task_seconds includes routing/embeddings and graph execution; graph_seconds excludes routing',
        'llm_call_count': 'Logical gen/agen invocations including failures and final node; '
                          'local HTTP request counts additionally include SDK retries',
        'cost_note': 'API cost only; local GPU/time cost is not measured',
    }
    writer = ResultWriter(result_file, metadata)
    status = 'completed'
    try:
        with torch.inference_mode():
            evaluate(router, dataset_test, writer, tasks, llms, reasonings,
                     args.prompt_file, args.max_tasks)
    except BaseException:
        status = 'interrupted'
        raise
    finally:
        metadata['finished_at'] = utc_now()
        writer.finish(status, excel=args.export_excel)
        logger.info(f"Results saved to {writer.path}")
    logger.info(f"Test accuracy: {writer.report['summary']['accuracy']}")