
#include <activations/dropout.h>
#include <utils/random.h>

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

Tensor<float>& DropoutLayer::forward(Tensor<float>& input)
{
    X = input;

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

    return Y;
}

Tensor<float>& DropoutLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

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

    return dX;
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
