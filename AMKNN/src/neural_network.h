#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <layers/neural_layers.h>
#include <activations/activation_layers.h>
#include <optimizers/optimizers.h>
#include <utils/convolution.h>
#include <data/dataset.h>
#include <data/list.h>
#include <data/array.h>
#include <common.h>

struct NeuralNetwork
{
    List<Parameter*> parameters;
    Array<NeuralLayer*> layers;
    Shape in_shape;

    NeuralNetwork(Shape _in_shape, std::vector<NeuralLayer*> _layers)
    {
        init(_in_shape, _layers);
    }
    NeuralNetwork() {}

    inline NeuralLayer* output_layer() { return layers[layers.size() - 1]; }
    inline NeuralLayer* input_layer() { return layers[0]; }

    void init(Shape _in_shape, std::vector<NeuralLayer*> _layers);
    void add(NeuralLayer* layer);

    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& output_grad);

    void setTrainable(bool option);

    void release();
    bool save(std::string filename);
    bool load(std::string filename);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

//----------------------------------------------
//  Loss Function
//----------------------------------------------
struct LossFunction
{
    Tensor<float> gradients;

    LossFunction(){}
    ~LossFunction(){}

    void init(int _grad_size);
    void release();
    virtual float evaluate(NeuralNetwork& network, DataSet& data, DataSet& labels) = 0;
    virtual Tensor<float>& gradient(NeuralNetwork& network, Tensor<float>& label, int batch_size) = 0;
};

struct MSELoss : LossFunction
{
    MSELoss() {};

    float evaluate(NeuralNetwork& network, DataSet& data, DataSet& labels);
    Tensor<float>& gradient(NeuralNetwork& network, Tensor<float>& label, int batch_size);
};
