
#include <layers/element_wise_linear.h>
#include <utils/random.h>

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

    BaseLayer::allocate(in_size, out_size);
}

EltwiseLinear::EltwiseLinear()
{
    setTrainable(true);
    weight_decay = 0;
    type = ELTWISE_LINEAR_LAYER;
}
EltwiseLinear::EltwiseLinear(float _weight_decay)
{
    setTrainable(true);
    weight_decay = _weight_decay;
    type = ELTWISE_LINEAR_LAYER;
}

Tensor<float>& EltwiseLinear::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = X[i] * A[i] + B[i];
    }

    return Y;
}

Tensor<float>& EltwiseLinear::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

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

    return dX;
}

void EltwiseLinear::save(std::ofstream& file)
{
    BaseLayer::save(file);
    file.write((char*)&weight_decay, sizeof(float));
}

void EltwiseLinear::load(std::ifstream& file)
{
    BaseLayer::load(file);
    file.read((char*)&weight_decay, sizeof(float));
}
