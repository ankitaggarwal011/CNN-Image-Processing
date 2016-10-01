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
cnn.ccd_diag('images/input.bmp', 'images/output6.png')
cnn.ccd_hor('images/input.bmp', 'images/output7.png')
cnn.ccd_hor_l('images/input.bmp', 'images/output8.png')
cnn.ccd_hor_r('images/input.bmp', 'images/output9.png')
cnn.ccd_ne('images/input.bmp', 'images/output10.png')
cnn.ccd_nw('images/input.bmp', 'images/output11.png')
cnn.ccd_se('images/input.bmp', 'images/output12.png')
cnn.ccd_sw('images/input.bmp', 'images/output13.png')
cnn.ccd_vert('images/input.bmp', 'images/output14.png')
cnn.ccd_vert_down('images/input.bmp', 'images/output15.png')
cnn.ccd_vert_top('images/input.bmp', 'images/output16.png')
cnn.smkiller('images/input.bmp', 'images/output17.png')
cnn.werosion('images/input.bmp', 'images/output18.png')

