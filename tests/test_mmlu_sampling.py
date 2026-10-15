from collections import Counter
from contextlib import redirect_stderr
import io
import random
import unittest

from MAR.Experiment.sampling import balanced_mmlu_test
import test_mmlu_results


class Dataset:
    split = 'test'
    def __init__(self, subjects=57, count=40):
        self.rows = [dict(subject=f'subject_{subject:02}', source_file=f'subject_{subject:02}_test.csv',
                          source_row=row, question=f'q{subject}/{row}', correct_answer='A')
                     for subject in range(subjects) for row in range(count)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]
    def record_to_input(self, row): return {'task': row['question']}
    def record_to_target_answer(self, row): return row['correct_answer']


class MMLUSamplingTests(unittest.TestCase):
    def test_twenty_five_per_subject_gives_1425_unique_test_questions(self):
        selected, manifest = balanced_mmlu_test(Dataset(), 25, 1234)
        self.assertEqual(len(selected), 1425)
        self.assertEqual(manifest['subject_count'], 57)
        self.assertEqual(set(Counter(selected[i]['subject'] for i in range(len(selected))).values()), {25})
        self.assertEqual(len(set(manifest['selected_task_ids'])), 1425)
        self.assertEqual(selected.split, 'test')
        self.assertEqual(selected.record_to_target_answer(selected[0]), 'A')
        self.assertEqual(selected.record_to_input(selected[0]), {'task': selected[0]['question']})
        for start in range(0, len(selected), 57):
            self.assertEqual(len({selected[i]['subject'] for i in range(start, start+57)}), 57)

    def test_selection_is_reproducible_independent_of_input_order_and_router_rng(self):
        dataset = Dataset()
        before = random.getstate()
        _, first = balanced_mmlu_test(dataset, 25, 1234)
        self.assertEqual(random.getstate(), before)
        dataset.rows.reverse()
        _, second = balanced_mmlu_test(dataset, 25, 1234)
        self.assertEqual(first['selection_sha256'], second['selection_sha256'])
        self.assertEqual(first['selected_task_ids'], second['selected_task_ids'])
        _, other = balanced_mmlu_test(dataset, 25, 999)
        self.assertNotEqual(first['selected_task_ids'], other['selected_task_ids'])

    def test_insufficient_subjects_empty_duplicates_or_non_test_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            balanced_mmlu_test(Dataset(count=2), 25, 1)
        with self.assertRaisesRegex(ValueError, 'empty'):
            balanced_mmlu_test(Dataset(subjects=0), 25, 1)
        dataset = Dataset(subjects=1)
        dataset.rows.append(dataset.rows[0].copy())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            balanced_mmlu_test(dataset, 25, 1)
        dataset.split = 'dev'
        with self.assertRaisesRegex(ValueError, 'test split'):
            balanced_mmlu_test(dataset, 25, 1)

    def test_cli_selects_balancing_without_truncating_a_subject(self):
        parse = test_mmlu_results.RunnerArgumentTests().parse
        args = parse(['--checkpoint', 'router.pth', '--samples_per_subject', '25'])
        self.assertEqual(args.samples_per_subject, 25)
        self.assertIsNone(args.max_tasks)
        self.assertIsNone(parse(['--allow_untrained']).samples_per_subject)
        for options in (['--samples_per_subject','0'], ['--samples_per_subject','-1'],
                        ['--samples_per_subject','25','--max_tasks','1400']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(['--allow_untrained', *options])
