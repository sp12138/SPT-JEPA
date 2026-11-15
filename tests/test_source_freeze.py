import tempfile
from pathlib import Path
import unittest

from experiments.robocasa_compare.source_freeze import freeze_with_overrides


class SourceFreezeTest(unittest.TestCase):
    def test_directory_and_file_aliases_are_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            shared=root/'shared';shared.mkdir();(shared/'old.py').write_bytes(b'original\n')
            (shared/'model.py').symlink_to('old.py')
            (source/'package').symlink_to(shared,target_is_directory=True)
            destination=root/'isolated'
            proof=freeze_with_overrides(source,destination,{'package/model.py':b'new\n'})
            self.assertEqual((shared/'old.py').read_bytes(),b'original\n')
            self.assertEqual((source/'package/model.py').read_bytes(),b'original\n')
            self.assertEqual((destination/'package/model.py').read_bytes(),b'new\n')
            self.assertFalse((destination/'package').is_symlink())
            self.assertFalse((destination/'package/model.py').is_symlink())
            self.assertTrue(proof['package/model.py']['private'])

    def test_readonly_relative_link_survives_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            dependency=root/'dependency';dependency.mkdir();(dependency/'x.py').write_bytes(b'dep\n')
            (source/'runtime').symlink_to('../dependency',target_is_directory=True)
            deeper=root/'deeper';deeper.mkdir();destination=deeper/'version'
            freeze_with_overrides(source,destination,{'model.py':b'local\n'})
            self.assertEqual((destination/'runtime/x.py').read_bytes(),b'dep\n')
            self.assertEqual((dependency/'x.py').read_bytes(),b'dep\n')

    def test_rejects_escape_and_existing_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            with self.assertRaises(ValueError):
                freeze_with_overrides(source,root/'bad',{'../escape.py':b'bad'})
            destination=root/'version';destination.mkdir()
            with self.assertRaises(FileExistsError):
                freeze_with_overrides(source,destination,{'model.py':b'bad'})
            self.assertFalse((root/'escape.py').exists())


if __name__=='__main__':unittest.main()
