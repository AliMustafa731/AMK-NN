#pragma once

#include <layers/base_layer.h>

/*
 * "Dropout" layer randomly drops (set to zero)
 * some of it's inputs with a propability "P"
 * to reduce overfitting
 */
struct DropoutLayer : BaseLayer
{
    /// array containing values that can be either (0) or (1)
    /// multiplied (element-wise) with the input to mask out places that has (0).
    Array<float> mask;

    /// the probability of inputs being kept (between 0.0 to 1.0)
    float P;

    DropoutLayer(float _P);
    DropoutLayer();

    /// @brief setup (input/output) shapes
    /// @param _in_shape : input shape
    void init(Shape _in_shape);

    /// @brief release the memory of "mask"
    void release();

    /*
     * Forward Pass :
     * drop some inputs randomly depending on the probability "P",
     * store the result in the output
     *
     * @param input : input Tensor
     * @return : output Tensor
     */
    Tensor<float>& forward(Tensor<float>& input);

    /*
     * Backward Pass :
     * caculate the gradients of the objective with respect to the "Input" and "Learned Parameters"
     * only masked inputs (multiplied by 1) recieve the gradients
     *
     * @param output_grad :
     * Tensor containing gradients of the objective with respect to the output Tensor
     * 
     * @return :
     * Tensor containing gradients with respect to "Input"
     */
    Tensor<float>& backward(Tensor<float>& output_grad);

    /// @brief save "P" hyper parameter to a file
    /// @param file : handle to a previously openned file
    void save(std::ofstream& file);

    /// @brief load "P" hyper parameter from a file
    /// @param file : handle to a previously openned file
    void load(std::ifstream& file);
};
