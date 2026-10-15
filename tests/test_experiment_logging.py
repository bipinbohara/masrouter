from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from loguru import logger
from MAR.Experiment.logging import experiment_file_log


class ExperimentFileLogTests(unittest.TestCase):
    def test_unique_files_capture_debug_and_keep_existing_sink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = logger.add(root / 'existing.log')
            try:
                with patch('MAR.Experiment.logging.LOG_DIR', root / 'tests'):
                    with experiment_file_log('gsm8k') as first:
                        logger.debug('generation details')
                    with experiment_file_log('gsm8k') as second:
                        logger.info('second test')
                self.assertNotEqual(first, second)
                self.assertIn('generation details', first.read_text())
                self.assertNotIn('second test', first.read_text())
                self.assertIn('second test', (root / 'existing.log').read_text())
            finally:
                logger.remove(existing)

    def test_startup_error_is_logged_and_sink_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('MAR.Experiment.logging.LOG_DIR', Path(directory)):
                with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                    with experiment_file_log('gsm8k') as path:
                        raise ValueError('bad checkpoint')
                saved = path.read_text()
                self.assertIn('bad checkpoint', saved)
                self.assertIn('Test run stopped', saved)
                logger.info('outside finished run')
                self.assertEqual(path.read_text(), saved)
