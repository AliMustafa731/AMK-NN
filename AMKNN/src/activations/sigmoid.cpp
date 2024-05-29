
#include <activations/sigmoid.h>
#include <cmath>

/*
 * A Layer that applies Non-linear "Sigmoid" Function To all of it's inputs.
 */

/// @brief setup (input/output) shapes
/// @param _in_shape : input shape
void SigmoidLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    BaseLayer::allocate(in_size, out_size);
}

SigmoidLayer::SigmoidLayer()
{
    type = SIGMOID_LAYER;
}

/// calculate the "Sigmoid" function
__forceinline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

/// an optimized method for calculating the derivative of "Sigmoid"
/// by using previously calculated values of "Sigmoid"
__forceinline float d_sigmoid_optimized(float x)
{
    // assumes (x) must contain Sigmoid(input)
    return x * (1.0f - x);
}

/*
 * Forward Pass :
 * apply non-linearity to the input, store the result in the output
 *
 * @param input : input Tensor
 * @return : output Tensor
 */
Tensor<float>& SigmoidLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = sigmoid(X[i]);
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
Tensor<float>& SigmoidLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = 0; i < in_size; i++)
    {
        dX[i] = d_sigmoid_optimized(Y[i]) * dY[i];
    }

    return dX;
}
