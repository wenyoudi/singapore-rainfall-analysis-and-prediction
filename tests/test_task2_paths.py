"""Account portability and write-boundary guards; no Spark installation required."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import task2_paths


class PathTests(unittest.TestCase):
    def test_accounts_are_isolated(self):
        for account in ['e1496655', 'e0969984', 'another_user']:
            with patch('task2_paths.pwd.getpwuid') as lookup:
                lookup.return_value.pw_name = account
                root = Path('/scratch') / account
                self.assertEqual(task2_paths.default_output(), str(root / 'task2/prepared'))
                self.assertEqual(task2_paths.output_path(root/'task2/new'), str(root/'task2/new'))
                for path in [root, root/'../other/data', Path('/tmp/data'), Path(str(root)+'_other/data')]:
                    with self.assertRaises(ValueError):
                        task2_paths.output_path(path)

    def test_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'own'
            other = Path(tmp)/'other'
            root.mkdir(); other.mkdir()
            (root/'escape').symlink_to(other, target_is_directory=True)
            with patch('task2_paths.scratch_root', return_value=root.resolve()):
                with self.assertRaises(ValueError):
                    task2_paths.output_path(root/'escape/data')

    def test_user_environment_does_not_change_owner(self):
        with patch('task2_paths.pwd.getpwuid') as lookup:
            lookup.return_value.pw_name = 'actual_account'
            with patch.dict('os.environ', {'USER': 'someone_else'}):
                self.assertEqual(task2_paths.scratch_root(), Path('/scratch/actual_account'))


if __name__ == '__main__':
    unittest.main()
