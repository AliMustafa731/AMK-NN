#pragma once

#include "layers/base_layer.h"

//----------------------------------------------
//  Average Pooling Layer
//----------------------------------------------
struct AvgPoolLayer : NeuralLayer
{
    Shape window, stride;

    AvgPoolLayer();
    AvgPoolLayer(Shape _window, Shape _stride);

    void init(Shape _in_shape);
    void release() {}
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
