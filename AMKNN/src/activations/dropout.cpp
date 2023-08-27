
#include "activations/dropout.h"
#include "utils/random.h"

void DropoutLayer::init(Shape _in_shape)
{
    out_size = in_size;
    out_shape = in_shape;

    mask.init(in_size);
}

void DropoutLayer::release()
{
    mask.release();
}

float* DropoutLayer::forward(float* input)
{
    X.data = input;

    for (int i = 0; i < X.size(); i++)
    {
        if (trainable)
        {
            if (random() < P)
            {
                mask[i] = 1.0f;
            }
            else
            {
                mask[i] = 0;
            }

            Y[i] = X[i] * mask[i] / P;
        }
        else
        {
            Y[i] = X[i] / P;
        }
    }

    return Y.data;
}

float* DropoutLayer::backward(float* d_output)
{
    dY.data = d_output;

    for (int i = 0; i < X.size(); i++)
    {
        if (trainable)
        {
            dX[i] = dY[i] * mask[i] / P;
        }
        else
        {
            dX[i] = dY[i] / P;
        }
    }

    return dX.data;
}

void DropoutLayer::save(std::ofstream& file)
{
    file.write((char*)&P, sizeof(float));
}

void DropoutLayer::load(std::ifstream& file)
{
    file.read((char*)&P, sizeof(float));
}

DropoutLayer::DropoutLayer(float _P)
{
    trainable = true;
    P = _P;
    type = DROPOUT_LAYER;
}
DropoutLayer::DropoutLayer()
{
    trainable = true;
    type = DROPOUT_LAYER;
}
