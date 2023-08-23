
#include "activations/activation_layers.h"

float* RelULeakLayer::forward(float* input)
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
            Y[i] = alpha * X[i];
        }
    }

    return Y.data;
}

float* RelULeakLayer::backward(float* d_output)
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
            dX[i] = alpha * dY[i];
        }
    }

    return dX.data;
}

void RelULeakLayer::save(std::ofstream& file)
{
    file.write((char*)&alpha, sizeof(float));
}

void RelULeakLayer::load(std::ifstream& file)
{
    file.read((char*)&alpha, sizeof(float));
}

RelULeakLayer::RelULeakLayer(float _alpha)
{
    alpha = _alpha;
    type = RELU_LEAK_LAYER;
}
RelULeakLayer::RelULeakLayer()
{
    type = RELU_LEAK_LAYER;
}

void RelULeakLayer::init(Shape _in_shape)
{
    out_size = in_size;
    out_shape = in_shape;
}
