"""Offline integrity tests. Run: python -m unittest discover -s tests -v"""

import ast
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('validate_submission', ROOT / 'scripts' / 'validate_submission.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.template = Path(self.directory.name) / 'template.csv'
        self.submission = Path(self.directory.name) / 'submission.csv'
        self.rows = [
            ['-32.1', '26.2', '01-02-2015', '100', '300', '20'],
            ['-32.2', '26.3', '02-02-2015', '120', '350', '25'],
        ]
        self.write_csv(self.template, [row[:3] + ['', '', ''] for row in self.rows])
        self.write_csv(self.submission, self.rows)

    def write_csv(self, path, rows, columns=None, encoding='utf-8'):
        with path.open('w', newline='', encoding=encoding) as handle:
            writer = csv.writer(handle)
            writer.writerow(validator.COLUMNS if columns is None else columns)
            writer.writerows(rows)

    def test_valid_submission(self):
        self.assertEqual(validator.validate_submission(self.submission, self.template), 2)

    def test_utf8_bom(self):
        self.write_csv(self.submission, self.rows, encoding='utf-8-sig')
        self.assertEqual(validator.validate_submission(self.submission, self.template), 2)

    def test_coordinate_rounding_is_allowed(self):
        self.rows[0][0] = '-32.1000002'
        self.write_csv(self.submission, self.rows)
        self.assertEqual(validator.validate_submission(self.submission, self.template), 2)

    def test_invalid_prediction_values(self):
        for value in ['', 'NaN', 'inf', '-inf', '-1', 'not a number']:
            with self.subTest(value=value):
                self.rows[0][3] = value
                self.write_csv(self.submission, self.rows)
                with self.assertRaises(ValueError):
                    validator.validate_submission(self.submission, self.template)

    def test_invalid_coordinate(self):
        self.rows[0][0] = 'NaN'
        self.write_csv(self.submission, self.rows)
        with self.assertRaises(ValueError):
            validator.validate_submission(self.submission, self.template)

    def test_changed_coordinate(self):
        self.rows[0][1] = '27.2'
        self.write_csv(self.submission, self.rows)
        with self.assertRaisesRegex(ValueError, 'ordering'):
            validator.validate_submission(self.submission, self.template)

    def test_changed_date(self):
        self.rows[0][2] = '03-02-2015'
        self.write_csv(self.submission, self.rows)
        with self.assertRaisesRegex(ValueError, 'Sample Date'):
            validator.validate_submission(self.submission, self.template)

    def test_reordered_rows(self):
        self.write_csv(self.submission, self.rows[::-1])
        with self.assertRaisesRegex(ValueError, 'ordering'):
            validator.validate_submission(self.submission, self.template)

    def test_missing_row(self):
        self.write_csv(self.submission, self.rows[:1])
        with self.assertRaisesRegex(ValueError, 'Row count'):
            validator.validate_submission(self.submission, self.template)

    def test_wrong_columns(self):
        self.write_csv(self.submission, self.rows, columns=validator.COLUMNS[::-1])
        with self.assertRaisesRegex(ValueError, 'expected columns'):
            validator.validate_submission(self.submission, self.template)

    def test_extra_field(self):
        self.write_csv(self.submission, [self.rows[0] + ['extra'], self.rows[1]])
        with self.assertRaisesRegex(ValueError, 'fields'):
            validator.validate_submission(self.submission, self.template)

    def test_empty_csv(self):
        self.submission.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'empty CSV'):
            validator.validate_submission(self.submission, self.template)

    def test_empty_template(self):
        self.write_csv(self.template, [])
        with self.assertRaisesRegex(ValueError, 'no prediction rows'):
            validator.validate_submission(self.submission, self.template)

    def test_cli_reports_failure(self):
        self.submission.unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            result = validator.main([str(self.submission), '--template', str(self.template)])
        self.assertEqual(result, 1)

    def test_cli_reports_success(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = validator.main([str(self.submission), '--template', str(self.template)])
        self.assertEqual(result, 0)


class RepositoryTests(unittest.TestCase):
    def test_all_python_files_parse(self):
        for path in list((ROOT / 'scripts').rglob('*.py')) + list((ROOT / 'tests').glob('*.py')):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

    def test_no_old_or_personal_paths(self):
        for path in (ROOT / 'scripts').rglob('*.py'):
            with self.subTest(path=path.name):
                source = path.read_text(encoding='utf-8-sig')
                self.assertNotIn('Jupyter Notebook Package', source)
                self.assertNotIn('C:\\Users\\', source)

    def test_paths_resolve_from_another_directory(self):
        # Evaluate only directory assignments, never import a training/extraction script.
        path_variables = {'BASE_DIR', 'ROOT', 'DATA', 'DATA_DIR', 'RAW_DIR',
                          'OUTPUT_DIR', 'EXTRACTED_DIR', 'OUT_DIR', 'SUBMISSION_DIR', 'LOG_DIR'}
        expected = {
            'RAW_DIR': ROOT / 'ey_challenge_notebooks',
            'DATA': ROOT / 'data' / 'extracted',
            'OUTPUT_DIR': ROOT / 'data' / 'extracted',
            'EXTRACTED_DIR': ROOT / 'data' / 'extracted',
            'OUT_DIR': ROOT / 'data' / 'extracted',
            'SUBMISSION_DIR': ROOT / 'submissions',
            'LOG_DIR': ROOT / 'experiment_logs',
        }
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for path in (ROOT / 'scripts').rglob('*.py'):
                    namespace = {'os': os, 'Path': Path, '__file__': str(path)}
                    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                    for node in tree.body:
                        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                                and isinstance(node.targets[0], ast.Name)
                                and node.targets[0].id in path_variables):
                            value = eval(compile(ast.Expression(node.value), str(path), 'eval'), namespace)
                            name = node.targets[0].id
                            namespace[name] = value
                            if name in {'BASE_DIR', 'ROOT'}:
                                continue
                            destination = expected.get(name)
                            if name == 'DATA_DIR':
                                destination = (ROOT / 'data' / 'extracted' if path.name in
                                               {'14_model_v7_ec.py', '17_model_v7b_ec.py'}
                                               else ROOT / 'ey_challenge_notebooks')
                            with self.subTest(path=path.name, variable=name):
                                self.assertEqual(Path(value).resolve(), destination)
            finally:
                os.chdir(old_cwd)

    def test_saved_best_submission(self):
        self.assertEqual(validator.validate_submission(
            ROOT / 'best_submission' / 'submission_v9.csv',
            ROOT / 'ey_challenge_notebooks' / 'submission_template.csv'), 200)


if __name__ == '__main__':
    unittest.main()
