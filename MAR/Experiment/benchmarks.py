"""Explicit held-out benchmark adapters; never carve training data out of test."""
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}. Provide the official evaluation file with --dataset_json.')
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BenchmarkDataset:
    split = 'test'
    def __init__(self, benchmark, rows, sources, protocol):
        self.benchmark, self.rows, self.sources, self.protocol = benchmark, rows, sources, protocol
        if not rows:
            raise ValueError(f'No {benchmark} test records found')
        ids = [str(row['task_id']) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate benchmark task IDs')
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]
    def record_to_input(self, row):
        return {'task': row['question']}
    def record_to_target_answer(self, row):
        return row['reference_answer']


def normalized(benchmark, path, index, question, reference, original, subject=None):
    return {'task_id': f'{benchmark}:{original.get("task_id", index)}',
            'source_file': str(path), 'source_row': index, 'question': question,
            'reference_answer': reference, 'subject': subject or benchmark, 'original': original}


def load_benchmark(benchmark, args):
    if benchmark == 'mmlu':
        from Datasets.mmlu_dataset import MMLUDataset
        return MMLUDataset('test')
    if benchmark == 'math':
        directory = Path(args.dataset_root) / 'test'
        paths = sorted(directory.rglob('*.json'))
        if not paths:
            raise FileNotFoundError(f'No MATH test JSON files found under {directory}')
        rows = []
        for index, path in enumerate(paths):
            item = json.loads(path.read_text(encoding='utf-8'))
            row = normalized(benchmark, path.relative_to(directory), index, item['problem'], item['solution'],
                             item, item.get('type', path.parent.name))
            row['task_id'] = 'math:' + str(path.relative_to(directory))
            rows.append(row)
        return BenchmarkDataset(benchmark, rows, {str(p): sha256(p) for p in paths}, 'Official MATH test directory')
    if benchmark == 'mbpp' and args.dataset_json is None:
        from Datasets.mbpp_dataset import MbppDataset
        frame = MbppDataset('test').df
        items = json.loads(frame.to_json(orient='records'))
        rows = [normalized(benchmark, 'google-research-datasets/mbpp/full/test', i, item['task'],
                           item.get('code'), item) for i, item in enumerate(items)]
        fingerprint = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
        return BenchmarkDataset(benchmark, rows, {'loaded_test_records': fingerprint}, 'Official Hugging Face MBPP full/test split')
    default = ROOT / ('Datasets/gsm8k/test.jsonl' if benchmark == 'gsm8k' else 'Datasets/humaneval/humaneval-py.jsonl')
    path = Path(args.dataset_json) if args.dataset_json else default
    items = read_jsonl(path)
    rows = []
    for index, item in enumerate(items):
        if item.get('split', 'test') != 'test':
            if benchmark == 'mbpp':
                continue
            raise ValueError(f'{path} contains non-test records; supply a test-only file')
        if benchmark == 'gsm8k':
            question, reference = item['question'], item['answer']
            protocol = 'User-designated official GSM8K test JSONL (no random partition)'
        elif benchmark == 'humaneval':
            question, reference = item['prompt'], item.get('canonical_solution')
            if not item.get('test') or not item.get('entry_point'):
                raise ValueError('HumanEval records require test and entry_point')
            protocol = 'Complete HumanEval evaluation suite (no official training split)'
        elif benchmark == 'mbpp':
            if 'split' not in item and not 11 <= int(item['task_id']) <= 510:
                continue
            tests = '\n'.join(item['test_list'])
            question = f"**Task**:\n```python\n{item['text']}\n```\nYour code should pass these tests:\n```python\n{tests}\n```"
            reference = item.get('code')
            protocol = 'MBPP test labels or official test task IDs 11–510'
        else:
            raise ValueError(f'Unknown benchmark: {benchmark}')
        rows.append(normalized(benchmark, path, index, question, reference, item))
    return BenchmarkDataset(benchmark, rows, {str(path): sha256(path)}, protocol)


def training_jsonl(benchmark, args):
    # Explicit separate input is required for HumanEval because it has no official train split.
    path = args.train_dataset_json or (ROOT / 'Datasets/gsm8k/train.jsonl' if benchmark == 'gsm8k' else None)
    if path is None:
        raise ValueError('HumanEval has no official train split; supply separate --train_dataset_json or use --mode test')
    rows = read_jsonl(path)
    evaluation = load_benchmark(benchmark, args)
    key = 'question' if benchmark == 'gsm8k' else 'prompt'
    seen = {row['original'][key] for row in evaluation.rows}
    if any(row[key] in seen for row in rows):
        raise ValueError('Training and evaluation questions overlap; supply disjoint training data')
    if benchmark == 'humaneval':
        test_ids = {row['original']['task_id'] for row in evaluation.rows}
        if any(row.get('task_id') in test_ids for row in rows):
            raise ValueError('HumanEval training and evaluation task IDs overlap')
    return rows


def extract_code(response):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', response, re.S | re.I)
    return '\n\n'.join(blocks).strip('\n') if blocks else response.strip('\n')


def score(benchmark, response, row, timeout=100):
    item = row['original']
    if benchmark == 'gsm8k':
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', response)
        explicit = re.findall(r'(?:final\s+answer|answer)\s*(?:is|:)\s*(.*)', response, re.I)
        text = boxed[-1] if boxed else explicit[-1] if explicit else response
        numbers = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
        predicted = numbers[-1].replace(',', '') if numbers else None
        reference = item['answer'].split('####')[-1].strip().replace(',', '')
        try:
            correct = predicted is not None and Decimal(str(predicted).replace(',', '')) == Decimal(reference)
        except (InvalidOperation, ValueError, TypeError):
            correct = False
        return {'predicted_answer': predicted or None, 'correct': correct, 'grading_method': 'numeric exact match'}
    if benchmark == 'math':
        from Datasets.math_dataset import MATH_get_predict, MATH_is_correct
        predicted = MATH_get_predict(response)
        return {'predicted_answer': predicted, 'correct': bool(MATH_is_correct(predicted, item['solution'])),
                'grading_method': 'Original MATH boxed-answer normalization/equivalence'}
    from MAR.Tools.coding.python_executor import PyExecutor
    code = extract_code(response)
    executor = PyExecutor()
    if benchmark == 'humaneval':
        entry = item['entry_point']
        # HumanEval completions may contain only the function body.
        if not re.search(rf'^\s*def\s+{re.escape(entry)}\s*\(', code, re.M):
            code = item['prompt'] + code
        correct = executor.evaluate(entry, code, item['test'], timeout=timeout)
        return {'predicted_answer': code, 'correct': bool(correct), 'grading_method': 'HumanEval check(entry_point), single-sample pass@1',
                'evaluation_tests': item['test'], 'entry_point': entry}
    tests = list(item['test_list'])
    if not tests:
        raise ValueError('MBPP test_list must not be empty')
    correct, feedback, state = executor.execute(code, tests, timeout=timeout)
    return {'predicted_answer': code, 'correct': bool(correct), 'grading_method': 'MBPP all assertions, single-sample pass@1',
            'evaluation_tests': tests, 'test_feedback': feedback, 'test_passed': list(state)}
