
#include <activations/sine.h>
#include <cmath>

void SineLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    NeuralLayer::allocate(in_size, out_size);
}

SineLayer::SineLayer()
{
    type = SINE_LAYER;
}

Tensor<float>& SineLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = sinf(X[i]);
    }

    return Y;
}

Tensor<float>& SineLayer::backward(Tensor<float>& output_grad)
{
    for (int i = 0; i < in_size; i++)
    {
        dX[i] = cosf(X[i]) * output_grad[i];
    }

    return dX;
}
