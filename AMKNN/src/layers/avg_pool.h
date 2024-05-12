#pragma once

#include <layers/base_layer.h>

//----------------------------------------------
//  Average Pooling Layer
//----------------------------------------------
struct AvgPoolLayer : BaseLayer
{
    Shape window, stride;

    AvgPoolLayer();
    AvgPoolLayer(Shape _window, Shape _stride);

    void init(Shape _in_shape);
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
