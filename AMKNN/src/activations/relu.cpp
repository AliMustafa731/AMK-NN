
#include "activations/relu.h"

float* RelULayer::forward(float* input)
{
    X.data = input;

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

    return Y.data;
}

float* RelULayer::backward(float* d_output)
{
    dY.data = d_output;

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

    return dX.data;
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
