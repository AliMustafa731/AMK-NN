
#include "layers/neural_layers.h"
#include "utils/utils.h"

void EltwiseLinear::init(Shape _in_shape)
{
    in_shape = _in_shape;
    out_shape = in_shape;
    in_size = in_shape.size();
    out_size = in_size;

    parameters.resize(2);
    parameters[0] = Parameter(in_size, weight_decay);
    parameters[1] = Parameter(in_size);

    A = parameters[0].values;
    dA = parameters[0].gradients;
    B = parameters[1].values;
    dB = parameters[1].gradients;

    for (int i = 0; i < B.size(); i++)
    {
        B[i] = random(-1.0f, 1.0f);
        dB[i] = 0;
    }
    for (int i = 0; i < A.size(); i++)
    {
        A[i] = random(-1.0f, 1.0f);
        dA[i] = 0;
    }

    setTrainable(true);
}

float* EltwiseLinear::forward(float* input)
{
    X.data = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = X[i] * A[i] + B[i];
    }

    return Y.data;
}

float* EltwiseLinear::backward(float* d_output)
{
    dY.data = d_output;

    if (trainable)
    {
        for (int i = 0; i < out_size; i++)
        {
            dB[i] += dY[i];
            dA[i] += X[i] * dY[i];
        }
    }

    for (int i = 0; i < out_size; i++)
    {
        dX[i] = A[i] * dY[i];
    }

    return dX.data;
}

void EltwiseLinear::save(std::ofstream& file)
{
    file.write((char*)&weight_decay, sizeof(float));
}

void EltwiseLinear::load(std::ifstream& file)
{
    file.read((char*)&weight_decay, sizeof(float));
}

EltwiseLinear::EltwiseLinear()
{
    trainable = true;
    weight_decay = 0;
    type = ELTWISE_LINEAR_LAYER;
}
EltwiseLinear::EltwiseLinear(float _weight_decay)
{
    trainable = true;
    weight_decay = _weight_decay;
    type = ELTWISE_LINEAR_LAYER;
}
