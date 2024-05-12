#pragma once

#include <layers/base_layer.h>

//----------------------------------------------
//  Element-wise linear sacle & offset
//----------------------------------------------
struct EltwiseLinear : BaseLayer
{
    Tensor<float> A, dA, B, dB;
    float weight_decay;

    EltwiseLinear();
    EltwiseLinear(float _weight_decay);

    void init(Shape _in_shape);
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
