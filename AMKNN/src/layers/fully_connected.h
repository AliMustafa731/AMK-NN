#pragma once

#include <layers/base_layer.h>

//----------------------------------------------
//  Fully Connected Layer
//----------------------------------------------
struct FullLayer : BaseLayer
{
    Tensor<float> W, dW, B, dB;
    float weight_decay;

    FullLayer();
    FullLayer(size_t _size, float _weight_decay = 0, Shape _out_shape = Shape(0, 0, 0));

    void init(Shape _in_shape);
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
