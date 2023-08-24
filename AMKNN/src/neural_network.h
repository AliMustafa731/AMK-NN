#pragma once

#include <iostream>
#include <vector>
#include <string>
#include "layers/neural_layers.h"
#include "activations/activation_layers.h"
#include "optimizers/optimizers.h"
#include "utils/utils.h"
#include "data/dataset.h"
#include "data/list.h"
#include "data/array.h"
#include "common.h"

struct NeuralNetwork
{
    List<Parameter*> parameters;
    Array<NeuralLayer*> layers;
    Array<float> loss_gradients;
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
    bool save(std::string filename);
    bool load(std::string filename);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

//----------------------------------------------
//  Mean Squared Error Loss Function
//----------------------------------------------
float MSELoss(NeuralNetwork* network, DataSet* data, DataSet* labels);

void MSELoss(NeuralNetwork* network, float* label, int batch_size);
