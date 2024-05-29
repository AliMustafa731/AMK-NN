
#include <activations/relu_leak.h>

/*
 * A Layer that applies Non-linear "Lekay Relu" Function To all of it's inputs.
 */

/// @brief setup (input/output) shapes
/// @param _in_shape : input shape
void RelULeakLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    BaseLayer::allocate(in_size, out_size);
}

/// constructors
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

/*
 * Forward Pass :
 * apply non-linearity to the input, store the result in the output
 *
 * @param input : input Tensor
 * @return : output Tensor
 */
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

/*
 * Backward Pass :
 * caculate the gradients of the objective with respect to the "Input" and "Learned Parameters"
 *
 * @param output_grad :
 * Tensor containing gradients of the objective with respect to the output Tensor
 * 
 * @return :
 * Tensor containing gradients with respect to "Input"
 */
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

/// @brief save "alpha" hyper parameter to a file
/// @param file : handle to a previously openned file
void RelULeakLayer::save(std::ofstream& file)
{
    BaseLayer::save(file);
    file.write((char*)&alpha, sizeof(float));
}

/// @brief load "alpha" hyper parameter from a file
/// @param file : handle to a previously openned file
void RelULeakLayer::load(std::ifstream& file)
{
    BaseLayer::load(file);
    file.read((char*)&alpha, sizeof(float));
}
