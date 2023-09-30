#pragma once

#include <layers/base_layer.h>

struct TanhLayer : NeuralLayer
{
    TanhLayer();

    void init(Shape _in_shape);
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);
};
