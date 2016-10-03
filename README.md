# PyCNN: Image Processing with Cellular Neural Networks in Python

[![Build Status](https://travis-ci.org/ankitaggarwal011/PyCNN.svg?branch=master)](https://travis-ci.org/ankitaggarwal011/PyCNN)

**Cellular Neural Networks (CNN)** [[wikipedia]](https://en.wikipedia.org/wiki/Cellular_neural_network) [[paper]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7600) are a parallel computing paradigm that was first proposed in 1988. Cellular neural networks are similar to neural networks, with the difference that communication is allowed only between neighboring units. Image Processing is one of its [applications](https://en.wikipedia.org/wiki/Cellular_neural_network#Applications). CNN processors were designed to perform image processing; specifically, the original application of CNN processors was to perform real-time ultra-high frame-rate (>10,000 frame/s) processing unachievable by digital processors.

This python library is the implementation of CNN for the application of **Image Processing**.

**Note**: The library has been **cited** in the research published on [Using Python and Julia for Efficient Implementation of Natural Computing and Complexity Related Algorithms](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7168488), look for the reference #19 in the references section. I'm glad that this library could be of help to the community.

**Note**: Cellular neural network (CNN) must not be confused with completely different convolutional neural network (ConvNet).

![alt text](http://www.isiweb.ee.ethz.ch/haenggi/CNN_web/CNN_figures/blockdiagram.gif "CNN Architecture")

As shown in the above diagram, imagine a control system with a feedback loop. f(x) is the sigmoidal kernel function for this system. The control and the feedback templates (coefficients) are configurable and controls the output of the system. Significant research had been done in determining the templates for common image processing techniques, these templates are published in this [Template Library](http://cnn-technology.itk.ppke.hu/Template_library_v4.0alpha1.pdf).

## Motivation

This is an extension of a demo at 14th Cellular Nanoscale Networks and Applications (CNNA) Conference 2014. I have written a blog post, available at [Image Processing in CNN with Python on Raspberry Pi](http://blog.ankitaggarwal.me/technology/image-processing-with-cellular-neural-networks-in-python-on-raspberry-pi).

## Dependencies

The library is supported for Python >= 2.7 and Python >= 3.3.

The python modules needed in order to use this library.
```
Pillow: 3.3.1
Scipy: 0.18.0
Numpy: 1.11.1 + mkl
```
Note: Scipy and Numpy can be installed on a Windows machines using binaries provided over [here](http://www.lfd.uci.edu/%7Egohlke/pythonlibs).

## Example 1

```sh
$ python example.py
```

#### OR

```python
from pycnn import pycnn

cnn = pycnn()
```

**Input:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/input1.bmp)

**Edge Detection:**

```python
cnn.edgedetection('images/input1.bmp', 'images/output1.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output1.png)

**Grayscale Edge Detection**

```python
cnn.grayscaleedgedetection('images/input1.bmp', 'images/output2.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output2.png)

**Corner Detection:**

```python
cnn.cornerdetection('images/input1.bmp', 'images/output3.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output3.png)

**Diagonal line Detection:**

```python
cnn.diagonallinedetection('images/input1.bmp', 'images/output4.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output4.png)

**Inversion (Logic NOT):**

```python
cnn.inversion('images/input1.bmp', 'images/output5.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/output5.png)

## Example 2

```sh
$ python example_lenna.py
```

#### OR

```python
from pycnn import pycnn

cnn = pycnn()
```

**Input:**

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna.gif)

**Edge Detection:**

```python
cnn.edgedetection('images/lenna.gif', 'images/lenna_edge.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna_edge.png)

**Diagonal line Detection:**

```python
cnn.diagonallinedetection('images/lenna.gif', 'images/lenna_diagonal.png')
```

![](https://raw.githubusercontent.com/ankitaggarwal011/PyCNN/master/images/lenna_diagonal.png)

## Usage

Import module

```python
from pycnn import pycnn
```

Initialize object

```python
cnn = pycnn()
```

```python
# name: name of image processing method (say, Edge detection); type: string
# inputimagelocation: location of the input image; type: string.
# outputimagelocation: location of the output image; type: string.
# tempA_A: control template; type: n x n list, e.g. 3 x 3, 5 x 5.
# tempB_B: feedback template; type: n x n list, e.g. 3 x 3, 5 x 5.
# initialcondition: initial condition, type: float.
# Ib_b: bias, type: float.
# t: time points for integration, type: ndarray. 
  # Note: Some image processing methods might need more time point samples than default.
  #       Display the output with each time point to see the evolution until the final convergence 
  #       to the output, looks pretty cool.
```

General image processing

```python
cnn.generaltemplates(name, inputimagelocation, outputimagelocation, tempA_A, tempB_B, 
                      initialcondition, Ib_b, t)
```

Edge detection

```python
cnn.edgedetection(inputimagelocation, outputimagelocation)
```

Grayscale edge detection

```python
cnn.grayscaleedgedetection(inputimagelocation, outputimagelocation)
```

Corner detection

```python
cnn.cornerdetection(inputimagelocation, outputimagelocation)
```

Diagonal line detection

```python
cnn.diagonallinedetection(inputimagelocation, outputimagelocation)
```

Inversion (Login NOT)

```python
cnn.inversion(inputimagelocation, outputimagelocation)
```

## License

[MIT License](https://github.com/ankitaggarwal011/PyCNN/blob/master/LICENSE)

## Contribute

Want to work on the project? Any kind of contribution is welcome!

Follow these steps:
- Fork the project.
- Create a new branch.
- Make your changes and write tests when practical.
- Commit your changes to the new branch.
- Send a pull request.
