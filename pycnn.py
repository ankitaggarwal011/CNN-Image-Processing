#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The MIT License (MIT)

Copyright (c) 2014 Ankit Aggarwal <ankitaggarwal011@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import print_function
import scipy.signal as sig
import scipy.integrate as sint
from PIL import Image as img
import numpy as np
import os.path
import warnings

SUPPORTED_FILETYPES = (
    'jpeg', 'jpg', 'png', 'tiff', 'gif', 'bmp',
)

warnings.filterwarnings('ignore')  # Ignore trivial warnings


class pycnn(object):

    def __init__(self):
        self._width = 0  # width (number of columns)
        self._height = 0  # height (number of rows)

    def f(self, x, t, Ib, Bu, tempA):
        x = x.reshape((self._height, self._width))
        dx = -x + Ib + Bu + sig.convolve2d(self.cnn(x), tempA, 'same')
        return dx.reshape(self._width * self._height)

    def cnn(self, x):
        return 0.5 * (abs(x + 1) - abs(x - 1))

    def validate(self, input_location):
        _, file_format = os.path.splitext(input_location)
        file_format = file_format.lstrip('.').lower()
        if not os.path.exists(input_location):
            raise IOError('File {} does not exist.'.format(input_location))
        elif not os.path.isfile(input_location):
            raise IOError('Path {} is not a file.'.format(input_location))
        elif file_format not in SUPPORTED_FILETYPES:
            raise Exception(
                '{} file type is not supported. Supported: {}'.format(
                    file_format, ', '.join(SUPPORTED_FILETYPES)
                )
            )

    # tempA: feedback template, tempB: control template
    def imageprocessing(self, inputlocation, outputlocation,
                        tempA, tempB, initialcondition, Ib, t):
        gray = img.open(inputlocation).convert('RGB')
        self._width, self._height = gray.size
        u = np.array(gray)
        u = u[:, :, 0]
        z0 = u * initialcondition
        Bu = sig.convolve2d(u, tempB, 'same')
        z0 = z0.flatten()
        z = self.cnn(sint.odeint(
            self.f, z0, t, args=(Ib, Bu, tempA), mxstep=1000))
        l = z[z.shape[0] - 1, :].reshape((self._height, self._width))
        l = l / (255.0)
        l = np.uint8(np.round(l * 255))
        # The direct vectorization was causing problems on Raspberry Pi.
        # In case anyone face a similar issue, use the below
        # loops rather than the above direct vectorization.
        # for i in range(l.shape[0]):
        #     for j in range(l.shape[1]):
        #         l[i][j] = np.uint8(round(l[i][j] * 255))
        l = img.fromarray(l).convert('RGB')
        l.save(outputlocation)

    # general image processing for given templates
    def generaltemplates(self,
                         name='Image processing',
                         inputlocation='',
                         outputlocation='output.png',
                         tempA_A=[[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]],
                         tempB_B=[[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]],
                         initialcondition=0.0,
                         Ib_b=0.0,
                         t=np.linspace(0, 10.0, num=2)):
        self.validate(inputlocation)
        print(name, 'initialized.')
        self.imageprocessing(inputlocation,
                             outputlocation,
                             np.array(tempA_A),
                             np.array(tempB_B),
                             initialcondition,
                             Ib_b,
                             t)
        print('Processing on image %s is complete' % (inputlocation))
        print('Result is saved at %s.\n' % (outputlocation))

    def edgedetection(self, inputlocation='', outputlocation='output.png'):
        name = 'Edge detection'
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
        Ib = -1.0
        # num refers to the number of samples of time points from start = 0 to
        # end = 10.0
        t = np.linspace(0, 10.0, num=2)
        # some image processing methods might require more time point samples.
        initialcondition = 0.0
        self.generaltemplates(
            name,
            inputlocation,
            outputlocation,
            tempA,
            tempB,
            initialcondition,
            Ib,
            t)

    def grayscaleedgedetection(self, inputlocation='',
                               outputlocation='output.png'):
        name = 'Grayscale edge detection'
        tempA = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
        Ib = -0.5
        t = np.linspace(0, 1.0, num=100)
        initialcondition = 0.0
        self.generaltemplates(
            name,
            inputlocation,
            outputlocation,
            tempA,
            tempB,
            initialcondition,
            Ib,
            t)

    def cornerdetection(self, inputlocation='', outputlocation='output.png'):
        name = 'Corner detection'
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 4.0, -1.0], [-1.0, -1.0, -1.0]]
        Ib = -5.0
        t = np.linspace(0, 10.0, num=10)
        initialcondition = 0.0
        self.generaltemplates(
            name,
            inputlocation,
            outputlocation,
            tempA,
            tempB,
            initialcondition,
            Ib,
            t)

    def diagonallinedetection(self, inputlocation='',
                              outputlocation='output.png'):
        name = 'Diagonal line detection'
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]
        Ib = -4.0
        t = np.linspace(0, 0.2, num=100)
        initialcondition = 0.0
        self.generaltemplates(
            name,
            inputlocation,
            outputlocation,
            tempA,
            tempB,
            initialcondition,
            Ib,
            t)

    def inversion(self, inputlocation='', outputlocation='output.png'):
        name = 'Inversion'
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        Ib = -2.0
        t = np.linspace(0, 10.0, num=100)
        initialcondition = 0.0
        self.generaltemplates(
            name,
            inputlocation,
            outputlocation,
            tempA,
            tempB,
            initialcondition,
            Ib,
            t)
