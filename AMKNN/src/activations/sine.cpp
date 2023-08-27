
#include "activations/sine.h"
#include <cmath>

float* SineLayer::forward(float* input)
{
    X.data = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = sinf(X[i]);
    }

    return Y.data;
}

float* SineLayer::backward(float* d_output)
{
    for (int i = 0; i < in_size; i++)
    {
        dX[i] = cosf(X[i]) * d_output[i];
    }

    return dX.data;
}

SineLayer::SineLayer()
{
    type = SINE_LAYER;
}

void SineLayer::init(Shape _in_shape)
{
    out_size = in_size;
    out_shape = in_shape;
}
