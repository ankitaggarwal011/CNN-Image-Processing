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

## start of module
import scipy.signal as sig
import scipy.integrate as sint
import Image as img
import numpy as np
import os.path


class cnnimg:
    def __init__(self):
        ##Supported Filetypes
        self.filetypes = ["jpeg", "jpg", "png", "tiff", "gif", "bmp"]
        return

    ##Helper Functions start
    def f(self, x, t, Ib, Bu, tempA):
        x = x.reshape((self.n, self.m))
        dx = -x + Ib + Bu + sig.convolve2d(self.cnn(x), tempA, 'same')
        return dx.reshape(self.m*self.n)

    def cnn(self, x):
      return 0.5*(abs(x + 1) - abs(x - 1))

    def isvalid(self, inputlocation):
        if not os.path.isfile(inputlocation):
            raise Exception("File does not exist")
        elif inputlocation.split(".")[1].lower() not in self.filetypes or "." not in inputlocation:
            raise Exception("Invalid File")
        else:
            return True

    def imageprocessing(self, inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t):
        gray = img.open(inputlocation).convert('RGB')
        self.m, self.n = gray.size
        u = np.array(gray)             #SYGNAŁ WEJŚCIOWY, NASZE U
        u = u[:,:,0] # konwersja do RGB utrzymuje trzy pola RGB w macierzy, 0 to intensity
        z0 = (u)*initialcondition #TODO: wat
        Bu = sig.convolve2d(u, tempB, 'same')
        z0 = z0.flatten()
        z = self.cnn(sint.odeint(self.f, z0, t, args=(Ib, Bu, tempA)))
        l = z[z.shape[0]-1,:].reshape((self.n, self.m))
        l = l/(255.0)
        l = np.uint8(np.round(l*255))
        l = img.fromarray(l).convert('RGB')
        l.save(outputlocation)
        print("Image Processing is successful.")
        return
    ## Helper Functions end

    ## Image Processing methods start

    #edge detection #working
    def edgedetection(self, inputlocation = "", outputlocation = "output.png"):
        if not self.isvalid(inputlocation):
            print ("Invalid Location. Please try again ... ")
            exit()
        print ("Edge detection is initialized ... ")
        tempA = np.zeros((3,3), dtype=float)
        tempB = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
        Ib = -1.0
        t = np.linspace(0, 10.0, num=2)
        initialcondition = 0.0
        self.imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        print ("Edge detection of image "+ inputlocation +" is complete and saved at " + outputlocation + '\n')
        return

    #grayscale edge detection #working
    def grayscaleedgedetection(self, inputlocation = "", outputlocation = "output.png"):
        if not self.isvalid(inputlocation):
            print ("Invalid Location. Please try again ... ")
            exit()
        print ("Grayscale edge detection is initialized ... ")
        tempA = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        tempB = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
        Ib = -0.5
        t = np.linspace(0, 1.0, num=100)
        initialcondition = 0.0
        self.imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        print ("Grayscale edge detection of image "+ inputlocation +" is complete and saved at " + outputlocation + '\n')
        return

    #corner #working
    def cornerdetection(self, inputlocation = "", outputlocation = "output.png"):
        if not self.isvalid(inputlocation):
            print ("Invalid Location. Please try again ... ")
            exit()
        print ("Corner detection is initialized ... ")
        tempA = np.array([[0.0, 0.0, 0.0], [0.0,1.0, 0.0], [0.0, 0.0, 0.0]])
        tempB = np.array([[-1.0, -1.0, -1.0], [-1.0, 4.0, -1.0], [-1.0, -1.0, -1.0]])
        Ib = -5.0
        t = np.linspace(0, 10.0, num=10)
        initialcondition = 0.0
        self.imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        print ("Corner detection of image "+ inputlocation +" is complete and saved at " + outputlocation + '\n')
        return

    #diagonal line detector #working
    def diagonallinedetection(self, inputlocation = "", outputlocation = "output.png"):
        if not self.isvalid(inputlocation):
            print ("Invalid Location. Please try again ... ")
            exit()
        print ("Diagonal line detection is initialized ... ")
        tempA = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        tempB = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
        Ib = -4.0
        t = np.linspace(0, 0.2, num=100)
        initialcondition = 0.0
        self.imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        print ("Diagonal line detection of image "+ inputlocation +" is complete and saved at " + outputlocation + '\n')
        return

    # logic NOT #inversion #working
    def inversion(self, inputlocation = "", outputlocation = "output.png"):
        if not self.isvalid(inputlocation):
            print ("Invalid Location. Please try again ... ")
            exit()
        print ("Inversion is initialized ... ")
        tempA = np.array([[0.0, 0.0, 0.0], [0.0,1.0, 0.0], [0.0, 0.0, 0.0]])
        tempB = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        Ib = -2.0
        t = np.linspace(0, 10.0, num=100)
        initialcondition = 0.0
        self.imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        print ("Inversion of image "+ inputlocation +" is complete and saved at " + outputlocation + '\n')
        return
    ## Image Processing methods end

    ## General Image Processing method with template input
    def generaltemplates(self, inputlocation = "", outputlocation = "output.png", tempA_A = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tempB_B = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], initialcondition = 0.0, Ib_b = 0.0, t = np.linspace(0, 10.0, num=2)):
        if not self.isvalid(inputlocation):
            print ("Invalid Location. Please try again ... ")
            exit()
        print ("Given templates application is initialized ... ")
        tempA=np.array(tempA_A)
        tempB=np.array(tempB_B)
        Ib = Ib_b
        self.imageprocessing(inputlocation, outputlocation, tempA, tempB, initialcondition, Ib, t)
        print ("Given templates applied to image "+ inputlocation +" is complete and saved at " + outputlocation + '\n')
        return

    ## end of module
