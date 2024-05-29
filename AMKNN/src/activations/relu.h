#pragma once

#include <layers/base_layer.h>

/*
 * A Layer that applies Non-linear "Relu" Function To all of it's inputs.
 */
struct RelULayer : BaseLayer
{
    RelULayer();

    /// @brief setup (input/output) shapes
    /// @param _in_shape : input shape
    void init(Shape _in_shape);

    /*
     * Forward Pass :
     * apply non-linearity to the input, store the result in the output
     *
     * @param input : input Tensor
     * @return : output Tensor
     */
    Tensor<float>& forward(Tensor<float>& input);

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
    Tensor<float>& backward(Tensor<float>& output_grad);
};
