## AMK-NN
Neural Networks Implementation in pure C++.  
The goal was to do everything from scratch without using any external libraries.  

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
