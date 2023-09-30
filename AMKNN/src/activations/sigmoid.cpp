
#include <activations/sigmoid.h>
#include <cmath>

void SigmoidLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    NeuralLayer::allocate(in_size, out_size);
}

SigmoidLayer::SigmoidLayer()
{
    type = SIGMOID_LAYER;
}

__forceinline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

__forceinline float d_sigmoid_optimized(float x)
{
    return x * (1.0f - x); // assumes (x) must contain Sigmoid(input)
}

Tensor<float>& SigmoidLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = sigmoid(X[i]);
    }

    return Y;
}

Tensor<float>& SigmoidLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = 0; i < in_size; i++)
    {
        dX[i] = d_sigmoid_optimized(Y[i]) * dY[i];
    }

    return dX;
}
