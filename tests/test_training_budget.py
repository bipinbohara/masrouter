"""Exercise actual runner budget/checkpoint control flow without LLM inference."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from MAR.Experiment.training import limit_training_batch, training_limit_reached


class TrainingControl(ast.NodeTransformer):
    """Remove inference/loss work, retaining each runner's actual budget loop."""
    def visit_Expr(self, node):
        return node if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'torch.save' else None

    def visit_Assign(self, node):
        names = {'current_batch', 'final_checkpoint', 'train_loader', 'train_batches'}
        return node if any(isinstance(t, ast.Name) and t.id in names for t in node.targets) else None

    def visit_AugAssign(self, node):
        return node if isinstance(node.target, ast.Name) and node.target.id in ('training_samples', 'training_updates') else None

    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call) or ast.unparse(node.iter.func) not in ('range', 'enumerate'):
            return None
        node = self.generic_visit(node)
        node.body = node.body or [ast.Pass()]
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        node.body = node.body or [ast.Pass()]
        return node


class TrainingBudgetTests(unittest.TestCase):
    def test_all_runner_parsers_default_to_fifty_and_reject_negative_limits(self):
        import argparse
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch
        from MAR.Experiment.runner import add_evaluation_arguments, validate_evaluation_arguments
        for benchmark in ('gsm8k', 'math', 'mbpp', 'humaneval', 'mmlu'):
            tree = ast.parse(Path(f'Experiments/run_{benchmark}.py').read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
            namespace = dict(argparse=argparse, add_evaluation_arguments=add_evaluation_arguments,
                             validate_evaluation_arguments=validate_evaluation_arguments)
            exec(compile(ast.Module(body=[function], type_ignores=[]), benchmark, 'exec'), namespace)
            with self.subTest(benchmark=benchmark):
                with patch('sys.argv', [benchmark, '--allow_untrained']):
                    self.assertEqual(namespace['parse_args']().max_train_samples, 50)
                with patch('sys.argv', [benchmark, '--allow_untrained', '--max_train_samples', '0']):
                    self.assertEqual(namespace['parse_args']().max_train_samples, 0)
                with patch('sys.argv', [benchmark, '--allow_untrained', '--max_train_samples', '-1']), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    namespace['parse_args']()

    def test_exact_global_budget_trims_final_batch(self):
        args = SimpleNamespace(max_train_samples=50)
        sizes, count = [], 0
        for epoch in range(5):
            if training_limit_reached(args, count):
                break
            for start in range(0, 1000, 16):
                batch = limit_training_batch(list(range(start, start+16)), args, count)
                if not batch:
                    break
                sizes.append(len(batch))
                count += len(batch)
        self.assertEqual(sizes, [16, 16, 16, 2])
        self.assertEqual(count, 50)

    def test_unlimited_and_small_batches(self):
        args = SimpleNamespace(max_train_samples=0)
        batch = [1, 2]
        self.assertIs(limit_training_batch(batch, args, 5000), batch)
        self.assertFalse(training_limit_reached(args, 5000))
        args.max_train_samples = 1
        self.assertEqual(limit_training_batch(batch, args, 0), [1])
        self.assertEqual(limit_training_batch(batch, args, 1), [])

    def test_all_actual_runner_loops_save_checkpoint_at_fifty_not_last_requested_epoch(self):
        import pandas as pd
        for benchmark in ('gsm8k', 'math', 'mbpp', 'humaneval', 'mmlu'):
            with self.subTest(benchmark=benchmark):
                tree = ast.parse(Path(f'Experiments/run_{benchmark}.py').read_text())
                loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                            and isinstance(n.target, ast.Name) and n.target.id in ('epoch', 'i_epoch'))
                loop = TrainingControl().visit(loop)
                args = SimpleNamespace(epochs=5, start_epoch=0, batch_size=16, max_train_samples=50)
                dataset = list(range(1000))
                class Dataset:
                    def record_to_input(self, row):
                        return {'task': str(row['id'])}
                    def record_to_target_answer(self, row):
                        return '1'
                frame = pd.DataFrame({'id': dataset})
                def dataloader(data, size, index):
                    return frame.iloc[index*size:(index+1)*size] if isinstance(data, Dataset) else data[index*size:(index+1)*size]
                saved = []
                namespace = dict(args=args, train_dataset=dataset, dataset_train=Dataset(),
                                 train_batches=63, num_batches=63, train_batch=40,
                                 dataloader=dataloader, limit_training_batch=limit_training_batch,
                                 training_limit_reached=training_limit_reached,
                                 training_samples=0, training_updates=0, final_checkpoint=None,
                                 MbppDataLoader=lambda data, batch_size, shuffle: [data[i:i+batch_size] for i in range(0,len(data),batch_size)],
                                 router=SimpleNamespace(state_dict=lambda: {'samples': namespace['training_samples']}),
                                 torch=SimpleNamespace(save=lambda state,path: saved.append((state,path))))
                module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
                exec(compile(module, benchmark, 'exec'), namespace)
                self.assertEqual(namespace['training_samples'], 50)
                self.assertEqual(namespace['training_updates'], 4)
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0][0], {'samples': 50})
                self.assertIn('epoch0', namespace['final_checkpoint'])
                self.assertEqual(saved[0][1], namespace['final_checkpoint'])