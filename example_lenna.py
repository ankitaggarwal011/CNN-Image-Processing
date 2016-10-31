#!/usr/bin/python
# -*- coding: utf-8 -*-
from PyCNN import PyCNN

# Initialize object
cnn = PyCNN()

# Perform respective image processing techniques on the given image

cnn.edgeDetection('images/lenna.gif', 'images/lenna_edge.png')
cnn.diagonalLineDetection('images/lenna.gif', 'images/lenna_diagonal.png')
