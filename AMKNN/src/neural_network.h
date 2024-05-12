#pragma once

#include <fstream>
#include <string>

#include <layers/base_layer.h>
#include <data/array.h>

struct NeuralNetwork : BaseLayer
{
    Array<BaseLayer*> layers;

    NeuralNetwork(Shape _in_shape)
    {
        init(_in_shape);
    }
    NeuralNetwork() {}

    inline BaseLayer* output_layer() { return layers[layers.size() - 1]; }
    inline BaseLayer* input_layer() { return layers[0]; }

    void init(Shape _in_shape);
    void add(BaseLayer* layer);

    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);

    void setTrainable(bool state);

    void release();
    bool save(std::string filename);
    bool load(std::string filename);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
