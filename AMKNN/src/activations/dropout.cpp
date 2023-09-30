
#include <activations/dropout.h>
#include <utils/random.h>

void DropoutLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    mask.init(in_size);

    NeuralLayer::allocate(in_size, out_size);
}

DropoutLayer::DropoutLayer(float _P)
{
    P = _P;
    type = DROPOUT_LAYER;
}
DropoutLayer::DropoutLayer()
{
    P = 0;
    type = DROPOUT_LAYER;
}

void DropoutLayer::release()
{
    mask.release();
    NeuralLayer::release();
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
    NeuralLayer::save(file);
    file.write((char*)&P, sizeof(float));
}

void DropoutLayer::load(std::ifstream& file)
{
    NeuralLayer::load(file);
    file.read((char*)&P, sizeof(float));
}
