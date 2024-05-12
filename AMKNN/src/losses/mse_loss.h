#pragma once

#include "loss_function.h"

struct MSELoss : LossFunction
{
    MSELoss() {};

    float evaluate(NeuralNetwork& network, Tensor<float>& data, Tensor<float>& labels);
    Tensor<float>& gradient(Tensor<float>& output, Tensor<float>& label, float batch_size);
};
