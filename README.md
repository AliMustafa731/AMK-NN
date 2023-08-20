# AMKNN
Neural Networks Implementation in pure C++
The goal was to do everything from scratch without using any external libraries.

# Demo
I also made a demo program (included in the code in files **program.cpp & program.h**) where I implemented **implicit neural representations** on images,
where I used a **fully-connected** neural network that take as input the **Normalized (x, y)** coordinates of a pixel, and output a single gray-scale color.
Since the coordinates are normalized, we can interpolate the **function (neural network after fitting to an image)** to get up-scaled image.
I also used the **Sine** function as a **non-linearity** (aka. Activation).

you can [download](https://mega.nz/file/BJcSFbwS#LtlwEECH3-bHDHCGQ0TsthSKivZF2HvNB5EUgdW-rJA) it.

inspired by : https://youtu.be/ZjxPPvqNp3k

Here's a screenshot of the demo :

![Demo](./screenshot.jpg)
