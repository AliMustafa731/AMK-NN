
#include <activations/tanh.h>
#include <cmath>

void TanhLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    BaseLayer::allocate(in_size, out_size);
}

TanhLayer::TanhLayer()
{
    type = TANH_LAYER;
}

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
