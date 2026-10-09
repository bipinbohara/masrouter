import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from MAR.Experiment.benchmarks import load_benchmark, score, training_jsonl
from MAR.Experiment.runner import add_evaluation_arguments, validate_evaluation_arguments
from MAR.Experiment.evaluation import evaluate
from MAR.Experiment.results import ResultWriter
from test_mmlu_results import FakeRouter


def args(**overrides):
    return SimpleNamespace(**dict(dict(dataset_json=None, dataset_root=None, train_dataset_json=None), **overrides))


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return str(path)


class DatasetTests(unittest.TestCase):
    def test_gsm8k_entire_test_file_no_partition_and_correct_numeric_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.jsonl'
            write_jsonl(path, [{'question': f'q{i}', 'answer': 'work\n#### -2.5'} for i in range(3)])
            dataset = load_benchmark('gsm8k', args(dataset_json=str(path)))
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0]['source_row'], 0)
            self.assertEqual(dataset[2]['source_row'], 2)
            self.assertTrue(score('gsm8k', r'The answer is \boxed{-2.5}', dataset[0])['correct'])
            self.assertIsNone(score('gsm8k', 'Unable to answer', dataset[0])['predicted_answer'])
            write_jsonl(path, [{'question': 'q', 'answer': '0', 'split': 'train'}])
            with self.assertRaisesRegex(ValueError, 'non-test'):
                load_benchmark('gsm8k', args(dataset_json=str(path)))

    def test_math_loads_only_test_and_retains_subject_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'test'):
                folder = root / split / 'algebra'
                folder.mkdir(parents=True)
                (folder / '1.json').write_text(json.dumps({'problem': split, 'solution': r'\boxed{2}', 'type': 'Algebra'}))
            dataset = load_benchmark('math', args(dataset_root=directory))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]['question'], 'test')
            self.assertEqual(dataset[0]['task_id'], 'math:algebra/1.json')
            self.assertTrue(score('math', r'Answer: \boxed{2}', dataset[0])['correct'])

    def test_mbpp_official_test_task_ids_only_and_assertions_really_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mbpp.jsonl'
            rows = [{'task_id': i, 'text': 'Write add(a,b)', 'code': 'def add(a,b): return a+b',
                     'test_list': ['assert add(1,2) == 3']} for i in (1,11,510,511,601)]
            write_jsonl(path, rows)
            dataset = load_benchmark('mbpp', args(dataset_json=str(path)))
            self.assertEqual([r['original']['task_id'] for r in dataset.rows], [11, 510])
            self.assertTrue(score('mbpp', '```python\ndef add(a,b): return a+b\n```', dataset[0])['correct'])
            self.assertFalse(score('mbpp', 'def add(a,b): return 0', dataset[0])['correct'])

    def test_mbpp_import_does_not_download_and_default_loads_only_test(self):
        import pandas as pd
        with patch('pandas.read_parquet') as read:
            import importlib
            import Datasets.mbpp_dataset as module
            importlib.reload(module)
            read.assert_not_called()
            read.return_value = pd.DataFrame([{'task_id': 11, 'text': 'q', 'code': 'pass', 'test_list': ['assert True']}])
            dataset = load_benchmark('mbpp', args())
            self.assertEqual(len(dataset), 1)
            self.assertEqual(read.call_count, 1)
            self.assertIn('/test-00000-of-00001.parquet', read.call_args.args[0])

    def test_humaneval_all_records_and_check_is_invoked_for_correct_and_wrong_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'humaneval.jsonl'
            rows = [{'task_id': f'HumanEval/{i}', 'prompt': 'def increment(x):\n', 'canonical_solution': '    return x+1\n',
                     'entry_point': 'increment', 'test': 'def check(candidate):\n    assert candidate(2) == 3\n'} for i in range(3)]
            write_jsonl(path, rows)
            dataset = load_benchmark('humaneval', args(dataset_json=str(path)))
            self.assertEqual(len(dataset), 3)
            self.assertTrue(score('humaneval', '    return x+1\n', dataset[0])['correct'])
            self.assertTrue(score('humaneval', '```python\ndef increment(x):\n    return x+1\n```', dataset[0])['correct'])
            self.assertFalse(score('humaneval', 'def increment(x):\n    return x-1', dataset[0])['correct'])

    def test_separate_training_rejects_overlapping_test_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            train, test = Path(directory)/'train.jsonl', Path(directory)/'test.jsonl'
            rows = [{'question': 'same', 'answer': '#### 2'}]
            write_jsonl(train, rows)
            write_jsonl(test, rows)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                training_jsonl('gsm8k', args(dataset_json=str(test), train_dataset_json=str(train)))
            write_jsonl(train, [{'question': 'other', 'answer': '#### 3'}])
            self.assertEqual(len(training_jsonl('gsm8k', args(dataset_json=str(test), train_dataset_json=str(train)))), 1)

    def test_all_new_adapters_use_same_per_task_export_without_tail_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configurations = {
                'gsm8k': [{'question': f'q{i}', 'answer': '#### 2'} for i in range(3)],
                'humaneval': [{'task_id': f'H/{i}', 'prompt': 'def f():\n', 'entry_point': 'f',
                              'test': 'def check(candidate):\n assert candidate()==2', 'canonical_solution':' return 2'} for i in range(3)],
                'mbpp': [{'task_id': i+11, 'text': 'q', 'test_list':['assert True'], 'code':'pass'} for i in range(3)],
            }
            for benchmark, rows in configurations.items():
                path = root / f'{benchmark}.jsonl'
                write_jsonl(path, rows)
                dataset = load_benchmark(benchmark, args(dataset_json=str(path)))
                writer = ResultWriter(root / f'{benchmark}.json', {})
                evaluate(FakeRouter(), dataset, writer, [], [], [], 'prompt',
                         scorer=lambda response, row: {'predicted_answer':'2', 'correct':True},
                         benchmark=benchmark, task_label=2 if benchmark != 'gsm8k' else 0)
                writer.finish(excel=True)
                report = json.loads(writer.path.read_text())
                self.assertEqual(len(report['tasks']), 3)
                self.assertEqual(report['tasks'][-1]['source_row'], 2)
                self.assertEqual(report['summary']['total_llm_calls'], 9)


class ArgumentTests(unittest.TestCase):
    def parser(self, benchmark):
        parser = argparse.ArgumentParser()
        parser.set_defaults(batch_size=1, epochs=1, max_agent=6, start_epoch=0)
        add_evaluation_arguments(parser, benchmark)
        return parser

    def test_every_benchmark_defaults_to_test_with_full_evaluation(self):
        for benchmark in ('gsm8k', 'math', 'mbpp', 'humaneval'):
            parser = self.parser(benchmark)
            options = parser.parse_args(['--allow_untrained'])
            validate_evaluation_arguments(parser, options, benchmark)
            self.assertEqual(options.mode, 'test')
            self.assertIsNone(options.max_tasks)

    def test_actual_four_runner_parsers_expose_test_only_defaults(self):
        import ast
        for benchmark in ('gsm8k', 'math', 'mbpp', 'humaneval'):
            source = Path(f'Experiments/run_{benchmark}.py').read_text()
            function = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
            namespace = {'argparse': argparse, 'add_evaluation_arguments': add_evaluation_arguments,
                         'validate_evaluation_arguments': validate_evaluation_arguments}
            exec(compile(ast.Module(body=[function], type_ignores=[]), benchmark, 'exec'), namespace)
            with patch('sys.argv', [benchmark, '--allow_untrained']):
                parsed = namespace['parse_args']()
            self.assertEqual(parsed.mode, 'test')
            self.assertIsNone(parsed.max_tasks)

    def test_humaneval_never_silently_uses_evaluation_suite_for_training(self):
        import io
        from contextlib import redirect_stderr
        parser = self.parser('humaneval')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_evaluation_arguments(parser, parser.parse_args(['--mode','train-test']), 'humaneval')
