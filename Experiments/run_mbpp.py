import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import re
import torch
from loguru import logger
import torch.nn.functional as F

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import get_llm_profiles
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging

from Datasets.mbpp_dataset import MbppDataset, MbppDataLoader

from MAR.Experiment.runner import add_evaluation_arguments, validate_evaluation_arguments, run_test
from MAR.Experiment.benchmarks import read_jsonl, training_jsonl

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
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on mbpp")
    parser.add_argument("--dataset_json", type=str, default=None, help="Official test JSONL file override")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16,help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Default 10.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--domain', type=str, default="mbpp",help="Domain (the same as dataset name), default 'mbpp'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/mbpp.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=400.0)
    parser.add_argument('--max_agent', type=int, default=6)
    parser.add_argument("--llm_tier", type=int, choices=[0, 1], default=None,
                        help="Use only local tier 0 (small) or 1 (large); default uses both")
    add_evaluation_arguments(parser, "mbpp")
    args = parser.parse_args()
    validate_evaluation_arguments(parser, args, "mbpp")
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == "test":
        run_test("mbpp", args)
        sys.exit(0)
    llms = get_llm_profiles(args.llm_tier)
    fix_random_seed(args.seed)
    train_dataset = MbppDataset('train')

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"mbpp_{current_time}.txt"
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=args.max_agent,device=device).to(device)
    if args.checkpoint:
        router.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    training_updates = 0
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    reasonings = reasoning_profile
    logger.info("Start training...")
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}",80*'-')
        total_solved, total_executed = (0, 0)
        train_loader = MbppDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"mbpp_router_epoch{epoch}_new.pth", map_location=device))
            continue
        for i_batch, current_batch in enumerate(train_loader):
            logger.info(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            queries = [item['task'] for item in current_batch]
            tests = [item['test_list'] for item in current_batch]
            task_labels = [2 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)
            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)
            
            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            utilities = []
            answers_loss = []
            is_solved_list = []
            pattern = r'```python.*```'
            for query, result, test, log_prob, cost in zip(queries, results, tests, log_probs, costs):
                match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved, _, _ = PyExecutor().execute(answer, test, timeout=100)
                else:
                    is_solved = 0
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * args.cost_rate
                utilities.append(utility)
                is_solved_list.append(is_solved)
                answer_loss = -log_prob * utility
                answers_loss.append(answer_loss)
            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            vae_loss = vae_loss.mean()
            is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [N, 1]
            adjust_loss = ((1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num) + 0.25 * is_solved_tensor *  agents_num).mean()
            
            loss = task_loss + answer_loss + vae_loss*0.001 # + adjust_loss
            loss.backward()
            optimizer.step()
            training_updates += 1
            accuracy = total_solved / total_executed

            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities:{utilities}")
        torch.save(router.state_dict(), f"mbpp_router_epoch{epoch}_new.pth")
    logger.info("End training...")
    final_checkpoint = f"mbpp_router_epoch{args.epochs-1}_new.pth" if training_updates or args.start_epoch else args.checkpoint
    run_test("mbpp", args, router=router, training_updates=training_updates,
             evaluated_checkpoint=final_checkpoint)
