from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from MAR.Experiment.evaluation import evaluate
from MAR.Experiment.results import ResultWriter
import test_mmlu_results as fixtures


def metadata(count=3):
    return {'started_at': 'original', 'seed': 1234, 'split': 'test',
            'dataset_size': count, 'requested_tasks': count,
            'test_sampling': {'selected_task_ids': [f'test:biology_test.csv:{i}' for i in range(count)]},
            'dataset_sha256': {'biology_test.csv': 'abc'}, 'prompt_sha256': 'prompt',
            'llm_profiles': [{'Name': 'local/model'}], 'evaluation_weights_source': 'random_initialization',
            'arguments': {'seed': 1234}, 'packages': {}}


def initial(path, count=2):
    writer = ResultWriter(path, metadata())
    evaluate(fixtures.FakeRouter(), fixtures.FakeDataset(), writer, [], [], [], 'prompt', max_tasks=count)
    writer.finish('interrupted')
    return writer


class ResumeTests(unittest.TestCase):
    def test_append_only_missing_tasks_rebuild_totals_events_json_and_excel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            original = initial(path)
            prefix = original.records_path.read_bytes()
            # Simulate a crash in event append after a complete task was saved.
            original.events_path.write_text('partial event')
            current = metadata()
            current['started_at'] = 'resumed'
            writer = ResultWriter(path, current, resume=True)
            self.assertEqual(writer.summary['total'], 2)
            router = fixtures.FakeRouter()
            evaluate(router, fixtures.FakeDataset(), writer, [], [], [], 'prompt')
            writer.finish(excel=True)
            self.assertEqual(router.seen, ['Question 2'])
            self.assertTrue(writer.records_path.read_bytes().startswith(prefix))
            result = json.loads(path.read_text())
            self.assertEqual(result['summary']['total'], 3)
            self.assertEqual(result['summary']['correct'], 3)
            self.assertEqual(len({r['task_id'] for r in result['tasks']}), 3)
            self.assertEqual(result['summary']['total_llm_calls'], 9)
            self.assertEqual(len(writer.events_path.read_text().splitlines()), 9)
            self.assertEqual(result['metadata']['started_at'], 'original')
            history = result['metadata']['resume_history'][-1]
            self.assertEqual(history['completed_before_resume'], 2)
            self.assertTrue((Path(history['backup_directory']) / 'experiment.tasks.jsonl').exists())
            self.assertTrue(path.with_suffix('.xlsx').exists())
            # Resuming a completed run makes no new calls or duplicate records.
            writer = ResultWriter(path, metadata(), resume=True)
            router = fixtures.FakeRouter()
            evaluate(router, fixtures.FakeDataset(), writer, [], [], [], 'prompt')
            writer.finish()
            self.assertEqual(router.seen, [])
            self.assertEqual(writer.summary['total'], 3)

    def test_settings_mismatch_refuses_without_changing_data(self):
        for key, value in [('seed', 9), ('dataset_sha256', {'biology_test.csv': 'changed'}),
                           ('llm_profiles', []), ('evaluated_checkpoint_sha256', 'different')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'experiment.json'
                original = initial(path)
                before = original.records_path.read_bytes()
                current = metadata()
                current[key] = value
                with self.assertRaisesRegex(ValueError, 'settings differ'):
                    ResultWriter(path, current, resume=True)
                self.assertEqual(original.records_path.read_bytes(), before)
                self.assertFalse((path.parent / '.resume_backups').exists())

    def test_partial_last_line_is_backed_up_and_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            original = initial(path)
            with original.records_path.open('ab') as stream:
                stream.write(b'{"task_id":')
            broken = original.records_path.read_bytes()
            writer = ResultWriter(path, metadata(), resume=True)
            self.assertEqual(writer.summary['total'], 2)
            backup = Path(writer.metadata['resume_history'][-1]['backup_directory'])
            self.assertEqual((backup / original.records_path.name).read_bytes(), broken)
            evaluate(fixtures.FakeRouter(), fixtures.FakeDataset(), writer, [], [], [], 'prompt')
            writer.finish()
            self.assertEqual(len(json.loads(path.read_text())['tasks']), 3)

    def test_keyboard_interrupted_last_task_is_retried_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            writer = ResultWriter(path, metadata())
            class Interrupt(fixtures.FakeRouter):
                def forward(self, *args, **kwargs):
                    if len(self.seen) == 1:
                        raise KeyboardInterrupt
                    return super().forward(*args, **kwargs)
            with self.assertRaises(KeyboardInterrupt):
                evaluate(Interrupt(), fixtures.FakeDataset(), writer, [], [], [], 'prompt')
            writer.finish('interrupted')
            self.assertEqual(writer.summary['total'], 2)
            resumed = ResultWriter(path, metadata(), resume=True)
            self.assertEqual(resumed.summary['total'], 1)
            self.assertIsNotNone(resumed.metadata['resume_history'][-1]['retried_interrupted_task_id'])
            router = fixtures.FakeRouter()
            evaluate(router, fixtures.FakeDataset(), resumed, [], [], [], 'prompt')
            resumed.finish()
            self.assertEqual(router.seen, ['Question 1', 'Question 2'])
            self.assertEqual(resumed.summary['total'], 3)

    def test_normal_task_errors_are_kept_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            writer = ResultWriter(path, metadata())
            evaluate(fixtures.FakeRouter(fail_index=0), fixtures.FakeDataset(), writer, [], [], [], 'prompt', max_tasks=1)
            writer.finish('interrupted')
            resumed = ResultWriter(path, metadata(), resume=True)
            router = fixtures.FakeRouter()
            evaluate(router, fixtures.FakeDataset(), resumed, [], [], [], 'prompt')
            resumed.finish()
            self.assertEqual(len(router.seen), 2)
            self.assertEqual(resumed.summary['failed'], 1)

    def test_changed_record_identity_or_bad_internal_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            original = initial(path)
            lines = original.records_path.read_text().splitlines()
            record = json.loads(lines[0]); record['task_id'] = 'wrong-task'
            original.records_path.write_text(json.dumps(record)+'\n'+lines[1]+'\n')
            with self.assertRaisesRegex(ValueError, 'balanced selection'):
                ResultWriter(path, metadata(), resume=True)
            original.records_path.write_text('{bad}\n'+lines[1]+'\n')
            with self.assertRaisesRegex(ValueError, 'Invalid task journal'):
                ResultWriter(path, metadata(), resume=True)

    def test_second_resume_process_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            initial(path)
            writer = ResultWriter(path, metadata(), resume=True)
            try:
                with self.assertRaisesRegex(RuntimeError, 'already writing'):
                    ResultWriter(path, metadata(), resume=True)
            finally:
                writer.finish('interrupted')

    def test_cli_requires_existing_output_and_test_mode(self):
        parse = fixtures.RunnerArgumentTests().parse
        args = parse(['--allow_untrained', '--resume', '--result_file', 'original/experiment.json'])
        self.assertTrue(args.resume)
        for options in (['--resume'], ['--resume','--mode','train-test','--result_file','x.json']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(['--allow_untrained', *options])
