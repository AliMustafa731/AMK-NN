#pragma once

#include <layers/base_layer.h>

struct SigmoidLayer : NeuralLayer
{
    SigmoidLayer();

    void init(Shape _in_shape);
    void release() {}
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};
