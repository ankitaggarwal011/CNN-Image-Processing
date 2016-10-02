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

warnings.filterwarnings("ignore") # Ignore trivial warnings

class pycnn:
    def __init__(self):
        self.filetypes = ["jpeg", "jpg", "png", "tiff", "gif", "bmp"] # Supported filetypes
        return

    def f(self, x, t, Ib, Bu, tempA):
        x = x.reshape((self.n, self.m))
        dx = -x + Ib + Bu + sig.convolve2d(self.cnn(x), tempA, 'same')
        return dx.reshape(self.m * self.n)

    def cnn(self, x):
      return 0.5 * (abs(x + 1) - abs(x - 1))

    def isvalid(self, inputlocation):
        if not os.path.isfile(inputlocation):
            raise Exception("File does not exist.")
        elif inputlocation.split(".")[1].lower() not in self.filetypes or "." not in inputlocation:
            raise Exception("Invalid File.")
        else:
            return True

    def imageprocessing(self, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t): #tempA: control template, tempB: feedback template
        gray = img.open(inputlocation).convert('RGB')
        self.m, self.n = gray.size
        u = np.array(gray)
        u = u[:, :, 0]
        z0 = u * initialcondition
        Bu = sig.convolve2d(u, tempB, 'same')
        z0 = z0.flatten()
        z = self.cnn(sint.odeint(self.f, z0, t, args=(Ib, Bu, tempA)))
        l = z[z.shape[0] - 1, :].reshape((self.n, self.m))
        l = l / (255.0)
        l = np.uint8(np.round(l * 255))
        # The direct vectorization was causing problems on Raspberry Pi.
        # In case anyone face a similar issue, use the below loops rather than the above direct vectorization.
        # for i in range(l.shape[0]):
        #     for j in range(l.shape[1]):
        #         l[i][j] = np.uint8(round(l[i][j] * 255))
        l = img.fromarray(l).convert('RGB')
        l.save(outputlocation)
        return

    ## general image processing for given templates
    def generaltemplates(self, name = "Image processing", inputlocation = "", outputlocation = "output.png", tempA_A = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tempB_B = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], initialcondition = 0.0, Ib_b = 0.0, t = np.linspace(0, 10.0, num=2)):
        if not self.isvalid(inputlocation):
            print("Invalid Location. Please try again.")
            exit()
        print(name, "initialized.")
        self.imageprocessing(inputlocation, outputlocation, np.array(tempA_A), np.array(tempB_B), initialcondition, Ib_b, t)
        print("Processing on image "+ inputlocation +" is complete and the result is saved at " + outputlocation + '.\n')
        return

    def edgedetection(self, inputlocation = "", outputlocation = "output.png"):
        name = "Edge detection"
        tempA = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
        Ib = -1.0
        t = np.linspace(0, 10.0, num=2)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def grayscaleedgedetection(self, inputlocation = "", outputlocation = "output.png"):
        name = "Grayscale edge detection"
        tempA = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
        Ib = -0.5
        t = np.linspace(0, 1.0, num=100)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def cornerdetection(self, inputlocation = "", outputlocation = "output.png"):
        name = "Corner detection"
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, -1.0, -1.0], [-1.0, 4.0, -1.0], [-1.0, -1.0, -1.0]] 
        Ib = -5.0
        t = np.linspace(0, 10.0, num=10)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def diagonallinedetection(self, inputlocation = "", outputlocation = "output.png"):
        name = "Diagonal line detection"
        tempA = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]
        Ib = -4.0
        t = np.linspace(0, 0.2, num=100)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def inversion(self, inputlocation = "", outputlocation = "output.png"):
        name = "Inversion"
        tempA = [[0.0, 0.0, 0.0], [0.0,1.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        Ib = -2.0
        t = np.linspace(0, 10.0, num=100)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    # Halftoning templates

    def halftoning_hlf3(self, inputlocation = "", outputlocation = "output.png"):
        name = "Halftoning HLF3"
        tempA = [[-0.07,-0.1,-0.07], [-0.1,1.03,-0.1], [-0.07,-0.1,-0.07]]
        tempB = [[0.07,0.1,0.07], [0.1, 0.32, 0.1], [0.07,0.1,0.07]]
        Ib = 0.0
        t = np.linspace(0, 10.0, num=100)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def halftoning_hlf5(self, inputlocation = "", outputlocation = "output.png"):
        name = "Halftoning HLF5"
        tempA = [[-0.0245, -0.07, -0.099, -0.07, -0.0245], 
                 [-0.07, -0.324, -0.46, -0.324, -0.07], 
                 [-0.099, -0.46, 1.05, -0.46, -0.099], 
                 [-0.07, -0.324, -0.46, -0.324, -0.07], 
                 [-0.0245, -0.07, -0.099, -0.07, -0.0245]]
        tempB = [[0.0245, 0.07, 0.099, 0.07, 0.0245], 
                 [0.07, 0.324, 0.46, 0.324, 0.07], 
                 [0.099, 0.46, 0.81, 0.46, 0.099], 
                 [0.07, 0.324, 0.46, 0.324, 0.07], 
                 [0.0245, 0.07, 0.099, 0.07, 0.0245]]
        Ib = 0.0
        t = np.linspace(0, 10.0, num=500)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def halftoning_hlf5kc(self, inputlocation = "", outputlocation = "output.png"):
        name = "Halftoning HLF5KC"
        tempA = [[-0.03, -0.086, -0.13, -0.086, -0.03], 
                 [-0.086, -0.359, -0.604, -0.359, -0.086], 
                 [-0.13, -0.604, 1.05, -0.604, -0.13], 
                 [-0.086, -0.359, -0.604, -0.359, -0.086], 
                 [-0.03, -0.086, -0.13, -0.086, -0.03]]
        tempB = [[0.0, 0.0, 0.068, 0.0, 0.0], 
                 [0.0, 0.355, 0.756, 0.355, 0.0], 
                 [0.068, 0.756, 2.122, 0.756, 0.068], 
                 [0.0, 0.355, 0.756, 0.355, 0.0], 
                 [0.0, 0.0, 0.068, 0.0, 0.0]]
        Ib = 0.0
        t = np.linspace(0, 10.0, num=1000)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    # Inverse Halftoning templates

    def halftoning_herring(self, inputlocation = "", outputlocation = "output.png"):
        name = "Inverse Halftoning Herring"
        tempA = [[0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0]]
        tempB = [[-0.16, -0.16, -0.16, -0.16, -0.16], 
                 [-0.16, -0.4, -0.4, -0.4, -0.16],
                 [-0.16, -0.4, 4.0, -0.4, -0.16],
                 [-0.16, -0.4, -0.4, -0.4, -0.16],
                 [-0.16, -0.16, -0.16, -0.16, -0.16]]
        Ib = 0.0
        t = np.linspace(0, 0.2, num=10)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def halftoning_invhlf3(self, inputlocation = "", outputlocation = "output.png"):
        name = "Inverse Halftoning HLF3"
        tempA = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        tempB = [[0.07,0.1,0.07], [0.1, 0.32, 0.1], [0.07,0.1,0.07]]
        Ib = 0.0
        t = np.linspace(0, 10.0, num=100)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return

    def halftoning_invhlf5(self, inputlocation = "", outputlocation = "output.png"):
        name = "Inverse Halftoning HLF5"
        tempA = [[0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0], 
                 [0.0, 0.0, 0.0, 0.0, 0.0]]
        tempB = [[0.0049, 0.014, 0.0198, 0.014, 0.0049], 
                 [0.014, 0.0648, 0.092, 0.0648, 0.014], 
                 [0.0198, 0.092, 0.162, 0.092, 0.0198], 
                 [0.014, 0.0648, 0.092, 0.0648, 0.014], 
                 [0.0049, 0.014, 0.0198, 0.014, 0.0049]]
        Ib = 0.0
        t = np.linspace(0, 10.0, num=500)
        initialcondition = 0.0
        self.generaltemplates(name, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        return  

