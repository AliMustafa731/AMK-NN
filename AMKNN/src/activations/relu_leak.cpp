
#include <activations/relu_leak.h>

void RelULeakLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    NeuralLayer::allocate(in_size, out_size);
}

RelULeakLayer::RelULeakLayer(float _alpha)
{
    alpha = _alpha;
    type = RELU_LEAK_LAYER;
}
RelULeakLayer::RelULeakLayer()
{
    alpha = 0;
    type = RELU_LEAK_LAYER;
}

Tensor<float>& RelULeakLayer::forward(Tensor<float>& input)
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
            Y[i] = alpha * X[i];
        }
    }

    return Y;
}

Tensor<float>& RelULeakLayer::backward(Tensor<float>& output_grad)
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
            dX[i] = alpha * dY[i];
        }
    }

    return dX;
}

void RelULeakLayer::save(std::ofstream& file)
{
    NeuralLayer::save(file);
    file.write((char*)&alpha, sizeof(float));
}

void RelULeakLayer::load(std::ifstream& file)
{
    NeuralLayer::load(file);
    file.read((char*)&alpha, sizeof(float));
}
