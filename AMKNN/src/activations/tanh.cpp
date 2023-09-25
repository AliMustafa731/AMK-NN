
#include <activations/tanh.h>
#include <cmath>

__forceinline float tanH(float x)
{
    return (2.0f / (1.0f + exp(x *-2.0f))) - 1.0f;
}
__forceinline float d_tanH_optimized(float x)
{
    return 1.0f - (x * x); // assumes (x) must contain Tanh(input)
}

Tensor<float>& TanhLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = tanH(X[i]);
    }

    return Y;
}

Tensor<float>& TanhLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = 0; i < in_size; i++)
    {
        dX[i] = d_tanH_optimized(Y[i]) * dY[i];
    }

    return dX;
}

TanhLayer::TanhLayer()
{
    type = TANH_LAYER;
}

void TanhLayer::init(Shape _in_shape)
{
    out_size = in_size;
    out_shape = in_shape;
}
