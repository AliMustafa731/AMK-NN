#pragma once

#include "layers/base_layer.h"

//-----------------------------------
//  Max Pooling Layer
//-----------------------------------
struct MaxPoolLayer : NeuralLayer
{
    Shape window, stride;
    Array<int> max_indices;

    MaxPoolLayer();
    MaxPoolLayer(Shape _window, Shape _stride);

    void init(Shape _in_shape);
    void release();
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
