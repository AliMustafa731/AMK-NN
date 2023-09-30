#pragma once

#include <layers/base_layer.h>

struct RelULeakLayer : NeuralLayer
{
    float alpha;

    RelULeakLayer(float _alpha);
    RelULeakLayer();

    void init(Shape _in_shape);
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
