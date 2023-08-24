# AMKNN
Neural Networks Implementation in pure C++
The goal was to do everything from scratch without using any external libraries.

## Demo 1 : Image Upscale
In this Demo, I implemented **implicit neural representations** on images,
where I used a **fully-connected** neural network that take as input the **Normalized (x, y)** coordinates of a pixel, and output a single gray-scale color.
Since the coordinates are normalized, we can interpolate the **function (neural network after fitting to an image)** to get up-scaled image.
I also used the **Sine** function as a **non-linearity** (aka. Activation).

you can [download](https://mega.nz/file/pM0UnBxZ#bbUbsSVTP682dloHIIiZceuk7KeqJ2vdmD0oJAcH7Ys) it.

inspired by : https://youtu.be/ZjxPPvqNp3k

Here's a screenshot of the demo :

![Demo](./screenshot.jpg)
