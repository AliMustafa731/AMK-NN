
#include <activations/relu.h>

Tensor<float>& RelULayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < X.size(); i++)
    {
        if (X[i] > 0)
        {
            Y[i] = X[i];
        }
        else
        {
            Y[i] = 0;
        }
    }

    return Y;
}

Tensor<float>& RelULayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = 0; i < X.size(); i++)
    {
        if (X[i] > 0)
        {
            dX[i] = dY[i];
        }
        else
        {
            dX[i] = 0;
        }
    }

    return dX;
}

RelULayer::RelULayer()
{
    type = RElU_LAYER;
}

void RelULayer::init(Shape _in_shape)
{
    out_size = in_size;
    out_shape = in_shape;
}
