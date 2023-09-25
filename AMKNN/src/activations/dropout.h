#pragma once

#include <layers/base_layer.h>

struct DropoutLayer : NeuralLayer
{
    Array<float> mask;  // contains values that can have either (0) or (1)
    float P;            // the probability of nodes being kept working

    DropoutLayer(float _P);
    DropoutLayer();

    void init(Shape _in_shape);
    void release();
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
