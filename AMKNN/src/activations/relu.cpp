
#include <activations/relu.h>

/*
 * A Layer that applies Non-linear "Relu" Function To all of it's inputs.
 */

/// @brief setup (input/output) shapes
/// @param _in_shape : input shape
void RelULayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    BaseLayer::allocate(in_size, out_size);
}

RelULayer::RelULayer()
{
    type = RElU_LAYER;
}

/*
 * Forward Pass :
 * apply non-linearity to the input, store the result in the output
 *
 * @param input : input Tensor
 * @return : output Tensor
 */
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
