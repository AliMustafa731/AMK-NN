#pragma once

#include <data/tensor.h>

struct NeuralNetwork;

struct LossFunction
{
    Tensor<float> gradients;

    LossFunction(){}
    ~LossFunction(){}

    void init(int _grad_size);
    void release();
    virtual float evaluate(NeuralNetwork& network, Tensor<float>& data, Tensor<float>& labels) = 0;
    virtual Tensor<float>& gradient(Tensor<float>& output, Tensor<float>& label, float batch_size) = 0;
};
