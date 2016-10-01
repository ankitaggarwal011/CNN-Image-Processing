#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycnn import pycnn

# Initialize object
cnn = pycnn()

# Perform respective image processing techniques on the given image

cnn.edgedetection('images/input.bmp', 'images/output1.png')
cnn.grayscaleedgedetection('images/input.bmp', 'images/output2.png')
cnn.cornerdetection('images/input.bmp', 'images/output3.png')
cnn.diagonallinedetection('images/input.bmp', 'images/output4.png')
cnn.inversion('images/input.bmp', 'images/output5.png')
cnn.diffus4('images/input.bmp', 'images/output6.png')

