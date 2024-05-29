#pragma once

#include <optimizers/optimizer.h>

//
// The "RMS Propagation" Optimizer
// for more info, see : https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/
//
struct RMSPropagation : Optimizer
{
    // hyper parameters
    float learning_rate, beta;

    RMSPropagation();
    RMSPropagation(float _learning_rate, float _beta);

    // tweak the parameters & zero thier gradients
    void update(Array<Parameter> &parameters);

    // save hyper parameters to a file
    void save(std::ofstream& file);

    // load hyper parameters from a file
    void load(std::ifstream& file);
};
