"""Regression tests for the legacy parser used by GSM8K training."""
import unittest

from Datasets.gsm8k_dataset import _remove_right_units, gsm_get_predict


class GSMTrainingParserTests(unittest.TestCase):
    def test_repeated_latex_units_do_not_abort_training_answer_extraction(self):
        response = r'The answer is 7 \text{ cookies}. Revenue increases by 28 \text{ dollars}.'
        # Reproduces the exact parser/float path in run_gsm8k's reward loop.
        predicted = gsm_get_predict(response)
        self.assertEqual(predicted, '7')
        self.assertTrue(float(predicted) == float('7'))
        self.assertFalse(float(predicted) == float('8'))

    def test_unit_removal_preserves_the_original_numeric_prefix(self):
        for text, expected in [(r'7 \text{ cookies}', '7 '),
                               (r'7 \text{ cookies} + 28 \text{ dollars}', '7 '),
                               ('7', '7')]:
            with self.subTest(text=text):
                self.assertEqual(_remove_right_units(text), expected)

    def test_plain_and_boxed_integer_answers_keep_existing_behavior(self):
        for response in ['The answer is 7.', r'\boxed{7}',
                         r'\boxed{7 \text{ cookies}}', 'Cookies Needed: 7']:
            with self.subTest(response=response):
                self.assertEqual(gsm_get_predict(response), '7')


if __name__ == '__main__':
    unittest.main()
