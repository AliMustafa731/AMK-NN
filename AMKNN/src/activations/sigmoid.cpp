
#include "activations/activation_layers.h"
#include <cmath>

__forceinline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

__forceinline float d_sigmoid_optimized(float x)
{
    return x * (1.0f - x); // assumes (x) must contain Sigmoid(input)
}

float* SigmoidLayer::forward(float* input)
{
    X.data = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = sigmoid(X[i]);
    }

    return Y.data;
}

float* SigmoidLayer::backward(float* d_output)
{
    dY.data = d_output;

    for (int i = 0; i < in_size; i++)
    {
        dX[i] = d_sigmoid_optimized(Y[i]) * dY[i];
    }

    return dX.data;
}

SigmoidLayer::SigmoidLayer()
{
    type = SIGMOID_LAYER;
}

void SigmoidLayer::init(Shape _in_shape)
{
    out_size = in_size;
    out_shape = in_shape;
}
