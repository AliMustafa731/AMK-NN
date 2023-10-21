## AMK-NN
Neural Networks Implementation in pure C++.  
The goal was to do everything from scratch without using any external libraries.  

## About
This Implementation is limited to **Sequential Neural Networks** only.  
  
Let **W** be a vector conatning the **Learned Parameters** of the neural network.  
Let **X** be the input vector.  
Let **Y(X, W)** be the output of the neural network.  
Then we need to define an Objective Function **E(Y, Y`)**, and then we optimize **W** to (minimize/maximize the Objective).  
A.K.A **Loss Function** - **Error Function**, The **Mean Squared Error (MSE)** Loss Function is used here.  
  
The Neural Network is composed of cascaded layers, that do the following :  
- Each layer takes the input (**X**) from the previous layer, And computes & passes the output (**Y**) to the next layer.  
- Each layer takes the gradients of the loss_wrt_output (**dE/dY**) from the next layer, And computes & passes the gradients of the loss_wrt_input (**dE/dX**) to the previous layer.  
- Each layer computes the gradients of the loss with respect to it's **Learned Parameters** (if it has, eg: wieghts, biases, ...) to later optimize them.  
  
Optimization of parameters is done by using **Gradient Descent** algorithm.   
  
![Sequential Neural Network](./sequential_network.jpg)

## Demo 1 : Image Upscale
**Status : Completed**  
**About :**  
  
Implementation of **implicit neural representations** on images.  
A **fully-connected** neural network takes the **Normalized (x, y)** coordinates of a pixel as input, and produces a single **Gray-scale color** as output.  
By evaluating the function (**Nueral network**) on every pixel's (x, y) of the image, we can obtain the full image.  
Since the function (**Nueral network**) is continuous, we can interpolate it to get up-scaled image.  
  
The **Sine** function was used as a **non-linearity** (a.k.a. Activation).  

download from [here](https://mega.nz/file/pM0UnBxZ#bbUbsSVTP682dloHIIiZceuk7KeqJ2vdmD0oJAcH7Ys).  

![Demo Screenshot](./screenshot.jpg)

## Demo 2 : Painter
**Status : In Progress**  
**About :**  
  
Applying **Gradient Descent** on **Parameterized** primative drawing functions such as **( drawLine(), drawCurve(), fillPolygon() )** to generate images.  
