
#include <activations/tanh.h>
#include <cmath>

/*
 * A Layer that applies Non-linear "Hyperbolic Tangent" Function To all of it's inputs.
 */

/// @brief setup (input/output) shapes
/// @param _in_shape : input shape
void TanhLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    BaseLayer::allocate(in_size, out_size);
}

TanhLayer::TanhLayer()
{
    type = TANH_LAYER;
}

/// calculate the "Tanh" function
__forceinline float tanH(float x)
{
    return (2.0f / (1.0f + exp(x *-2.0f))) - 1.0f;
}

/// an optimized method for calculating the derivative of "Tanh"
/// by using previously calculated values of "Tanh"
__forceinline float d_tanH_optimized(float x)
{
    // assumes (x) must contain Tanh(input)
    return 1.0f - (x * x);
}

/*
 * Forward Pass :
 * apply non-linearity to the input, store the result in the output
 *
 * @param input : input Tensor
 * @return : output Tensor
 */
Tensor<float>& TanhLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < out_size; i++)
    {
        Y[i] = tanH(X[i]);
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
Tensor<float>& TanhLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = 0; i < in_size; i++)
    {
        dX[i] = d_tanH_optimized(Y[i]) * dY[i];
    }

    return dX;
}
