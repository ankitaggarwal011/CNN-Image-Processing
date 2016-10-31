import os
import unittest
import tempfile
import shutil

from pycnn import PyCNN

BASE_DIR = os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__)),
    '..',
))


class TestMiscFunctions(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.cnn = PyCNN()
        self.exits_file_name = os.path.join(self.tempdir, 'exist.bmp')
        open(self.exits_file_name, 'w').close()
        self.not_supported_file = os.path.join(self.tempdir, 'exist.jp2')
        open(self.not_supported_file, 'w').close()

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_is_valid(self):
        self.cnn.validate(self.exits_file_name)
        with self.assertRaisesRegexp(IOError, 'does not exist'):
            self.cnn.validate('/not/exist/filename')
        with self.assertRaisesRegexp(IOError, 'is not a file'):
            self.cnn.validate(self.tempdir)
        with self.assertRaisesRegexp(Exception, 'is not supported'):
            self.cnn.validate(self.not_supported_file)
