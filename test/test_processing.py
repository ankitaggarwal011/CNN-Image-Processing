import os
import tempfile
import unittest
import filecmp

from pycnn import PyCNN

IMAGE_DIR = os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__)),
    '..',
    'images',
))


class TestProcessing(unittest.TestCase):

    def setUp(self):
        self.cnn = PyCNN()
        self.input1 = os.path.join(IMAGE_DIR, 'input1.bmp')
        self.input3 = os.path.join(IMAGE_DIR, 'input3.bmp')
        self.output = os.path.join(
            tempfile.gettempdir(),
            'cnn_output.png',
        )

    def tearDown(self):
        if os.path.exists(self.output):
            os.remove(self.output)

    def test_edgeDetection(self):
        self.cnn.edgeDetection(self.input1, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output1.png'),
        ))

    def test_grayScaleEdgeDetection(self):
        self.cnn.grayScaleEdgeDetection(self.input1, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output2.png'),
        ))

    def test_cornerDetection(self):
        self.cnn.cornerDetection(self.input1, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output3.png'),
        ))

    def test_diagonalLineDetection(self):
        self.cnn.diagonalLineDetection(self.input1, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output4.png'),
        ))

    def test_inversion(self):
        self.cnn.inversion(self.input1, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output5.png'),
        ))

    def test_optimalEdgeDetection(self):
        self.cnn.optimalEdgeDetection(self.input3, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'output6.png'),
        ))


class TestProcessingLena(unittest.TestCase):

    def setUp(self):
        self.cnn = PyCNN()
        self.input = os.path.join(IMAGE_DIR, 'lenna.gif')
        self.output = os.path.join(
            tempfile.gettempdir(),
            'cnn_output.png',
        )

    def test_edgeDetection(self):
        self.cnn.edgeDetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'lenna_edge.png'),
        ))

    def test_diagonalLineDetection(self):
        self.cnn.diagonalLineDetection(self.input, self.output)
        self.assertTrue(filecmp.cmp(
            self.output, os.path.join(IMAGE_DIR, 'lenna_diagonal.png'),
        ))
