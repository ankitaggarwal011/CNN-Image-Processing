#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycnn import pycnn

# Initialize object
cnn = pycnn()

# Perform respective image processing techniques on the given image

cnn.edgedetection('images/input1.bmp', 'images/output1.png')
cnn.grayscaleedgedetection('images/input1.bmp', 'images/output2.png')
cnn.cornerdetection('images/input1.bmp', 'images/output3.png')
cnn.diagonallinedetection('images/input1.bmp', 'images/output4.png')
cnn.inversion('images/input1.bmp', 'images/output5.png')
