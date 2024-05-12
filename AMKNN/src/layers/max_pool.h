#pragma once

#include <layers/base_layer.h>

//-----------------------------------
//  Max Pooling Layer
//-----------------------------------
struct MaxPoolLayer : BaseLayer
{
    Shape window, stride;
    Array<int> max_indices;

    MaxPoolLayer();
    MaxPoolLayer(Shape _window, Shape _stride);

    void init(Shape _in_shape);
    void release();
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
