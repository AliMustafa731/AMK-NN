#pragma once

#include <iostream>
#include <vector>
#include <string>
#include "layers/neural_layers.h"
#include "activations/activation_layers.h"
#include "optimizers.h"
#include "data/utils.h"
#include "data/data.h"
#include "common.h"

struct NeuralNetwork
{
    std::vector<NeuralLayer*> layers;
    std::vector<Parameter*> parameters;
    std::vector<float> loss_gradients;
    Optimizer* optimizer;
    Shape in_shape;

    NeuralNetwork(Shape _in_shape, std::vector<NeuralLayer*> _layers, Optimizer* _optimizer)
    {
        init(_in_shape, _layers, _optimizer);
    }
    NeuralNetwork() {}

    NeuralLayer* output_layer() { return layers[layers.size() - 1]; }
    NeuralLayer* input_layer() { return layers[0]; }

    void init(Shape _in_shape, std::vector<NeuralLayer*> _layers, Optimizer* _optimizer);
    void add(NeuralLayer* layer);

    float* forward(float* input);
    float* backward(float* d_output);
    float* backward();

    void set_trainable(bool option);

    void release();
    void save(std::string filename);
    void load(std::string filename);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

//----------------------------------------------
//  Mean Squared Error Loss Function
//----------------------------------------------
float MSELoss(NeuralNetwork* network, DataSet* data, DataSet* labels);

void MSELoss(NeuralNetwork* network, float* label, int batch_size);
