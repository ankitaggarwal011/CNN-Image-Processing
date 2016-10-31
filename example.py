#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycnn import PyCNN

# Initialize object
cnn = PyCNN()

# Perform respective image processing techniques on the given image

cnn.edgeDetection('images/input1.bmp', 'images/output1.png')
cnn.grayScaleEdgeDetection('images/input1.bmp', 'images/output2.png')
cnn.cornerDetection('images/input1.bmp', 'images/output3.png')
cnn.diagonalLineDetection('images/input1.bmp', 'images/output4.png')
cnn.inversion('images/input1.bmp', 'images/output5.png')
cnn.optimalEdgeDetection('images/input3.bmp', 'images/output6.png')
