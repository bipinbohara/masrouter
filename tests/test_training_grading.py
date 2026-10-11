import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from MAR.Experiment.training import grade_training_answer


class TrainingGradingTests(unittest.TestCase):
    def test_empty_and_nonstring_answers_are_incorrect_for_all_benchmarks(self):
        for name in ('mmlu', 'gsm8k', 'math', 'mbpp', 'humaneval'):
            for response in ('', '  ', None, []):
                with self.subTest(benchmark=name, response=response):
                    grade = grade_training_answer(name, response, {})
                    self.assertFalse(grade['correct'])
                    self.assertIsNone(grade['predicted_answer'])

    def test_mmlu_no_indexing_and_correct_option_extraction(self):
        for response, expected in [(r'The answer is \boxed{B}.', True),
                                   ('Final answer: B', True), ('B', True),
                                   ('Answer: A', False), ('No answer available', False)]:
            self.assertEqual(grade_training_answer('mmlu', response, {'answer': 'B'})['correct'], expected)

    def test_gsm_numeric_rewards_match_test_grading_without_false_zero_answers(self):
        for response, reference, expected in [(r'\boxed{-2.5}', '-2.5', True),
                                               ('Final answer: 1,000', '1000', True),
                                               ('No answer available', '0', False),
                                               ('Final answer: 7', '8', False)]:
            self.assertEqual(grade_training_answer('gsm8k', response, {'answer': '#### '+reference})['correct'], expected)

    def test_parser_or_executor_errors_are_incorrect_and_reported(self):
        for benchmark in ('math', 'mbpp', 'humaneval'):
            with patch('MAR.Experiment.benchmarks.score', side_effect=SyntaxError('invalid output')):
                grade = grade_training_answer(benchmark, 'malformed', {})
            self.assertFalse(grade['correct'])
            self.assertIn('SyntaxError', grade['grading_error'])
        with patch('MAR.Experiment.benchmarks.score', side_effect=ImportError('missing dependency')):
            with self.assertRaises(ImportError):
                grade_training_answer('math', 'answer', {})
        with patch('MAR.Experiment.benchmarks.score', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                grade_training_answer('math', 'answer', {})

    def test_code_checkers_receive_full_record_and_timeout(self):
        for benchmark, record in [('mbpp', {'test_list': ['assert f() == 1']}),
                                  ('humaneval', {'entry_point': 'f', 'prompt': 'def f():', 'test': 'def check(candidate): ...'})]:
            with patch('MAR.Experiment.benchmarks.score', return_value={'correct': True}) as scorer:
                self.assertTrue(grade_training_answer(benchmark, 'code', record, 12)['correct'])
                scorer.assert_called_once_with(benchmark, 'code', {'original': record}, 12)

    def test_actual_mmlu_training_grade_statements_handle_unparseable_response(self):
        tree = ast.parse(Path('Experiments/run_mmlu.py').read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == 'log_prob' for e in n.target.elts))
        statements = [n for n in loop.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('grade', 'predict_answer', 'is_solved') for t in n.targets)]
        namespace = dict(result='', answer='B', grade_training_answer=grade_training_answer)
        exec(compile(ast.Module(body=statements, type_ignores=[]), 'mmlu_training', 'exec'), namespace)
        self.assertFalse(namespace['is_solved'])
        self.assertIsNone(namespace['predict_answer'])