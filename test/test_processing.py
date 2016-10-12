import os
import tempfile
import unittest
import filecmp

from pycnn import pycnn

IMAGE_DIR = os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__)),
    '..',
    'images',
))


class TestProcessing(unittest.TestCase):

    def setUp(self):
        self.cnn = pycnn()
        self.input = os.path.join(IMAGE_DIR, 'input1.bmp')
        self.output = os.path.join(
            tempfile.gettempdir(),
            'cnn_output.png',
        )

    def tearDown(self):
        if os.path.exists(self.output):
            os.remove(self.output)

    def test_edgedetection(self):
        self.cnn.edgedetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output1.png'),
        ))

    def test_grayscaleedgedetection(self):
        self.cnn.grayscaleedgedetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output2.png'),
        ))

    def test_cornerdetection(self):
        self.cnn.cornerdetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output3.png'),
        ))

    def test_diagonallinedetection(self):
        self.cnn.diagonallinedetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output4.png'),
        ))

    def test_inversion(self):
        self.cnn.inversion(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output5.png'),
        ))


class TestProcessingLena(unittest.TestCase):

    def setUp(self):
        self.cnn = pycnn()
        self.input = os.path.join(IMAGE_DIR, 'lenna.gif')
        self.output = os.path.join(
            tempfile.gettempdir(),
            'cnn_output.png',
        )

    def test_edgedetection(self):
        self.cnn.edgedetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'lenna_edge.png'),
        ))

    def test_diagonallinedetection(self):
        self.cnn.diagonallinedetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'lenna_diagonal.png'),
        ))
