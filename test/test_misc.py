import os
import unittest

from pycnn import pycnn

BASE_DIR = os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__)),
    '..',
))


class TestMiscFunctions(unittest.TestCase):

    def setUp(self):
        self.cnn = pycnn()

    def test_is_valid(self):
        input_file = os.path.join(
            BASE_DIR, 'images', 'input1.bmp',
        )
        result = self.cnn.isvalid(input_file)
        self.assertTrue(result)
