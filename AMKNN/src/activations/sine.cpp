
#include <activations/sine.h>
#include <cmath>

/*
 * A Layer that applies the Periodic Non-linear "Sine" Function To all of it's inputs.
 */

/// @brief setup (input/output) shapes
/// @param _in_shape : input shape
void SineLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    BaseLayer::allocate(in_size, out_size);
}

SineLayer::SineLayer()
{
    type = SINE_LAYER;
}

/*
 * Forward Pass :
 * apply non-linearity to the input, store the result in the output
 *
 * @param input : input Tensor
 * @return : output Tensor
 */
Tensor<float>& SineLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = sinf(X[i]);
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
Tensor<float>& SineLayer::backward(Tensor<float>& output_grad)
{
    for (int i = 0; i < in_size; i++)
    {
        dX[i] = cosf(X[i]) * output_grad[i];
    }

    return dX;
}
