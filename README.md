# conv2d_transpose
How Tensorflow's conv2d_transpose works

This is a simple tutorial to show how Tensorflow's conv2d_transpose works. I obtain the conv2d_transpose() output using conv2d_backprop() and conv2d() functions by performing deconvolution from the activations of the first convolutional layer. The results are that conv2d_transpose() transposes the filter's weights and flips them by 180 degrees. 

I test this using a CNN on the MNIST dataset which obtains 99,2% test accuracy. The checkpoint of the already trained network is given.

The details of the CNN are:
1) Input layer: the inputs are MNIST images of 28x28x1
2) Conv1 layer: convolutional layer with 32 filters of 7x7, input = 1x28x28x1 , output = 1x28x28x32 , weights = 7x7x1x32
3) Conv2 layer: conv layer with 64 filters, input= 32x28x28x1 , output = 64x28x28x1 , weights = 3x3x32x64
4) FC layer with 1024 units
5) FC layer with 10 units
6) Output = 10 scalars used to classify into the 10 MNIST's classes

To run this test I used Tensorflow 1.0 on Ubuntu 16.04, numpy 1.13.0, python 3.4 

To use it:
- Check requirements ( not sure if it will work with other versions )
- Download all the files and place them into a single folder
- Run the test.py script
- An image should compare in the folder
