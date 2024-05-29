#pragma once

#include <optimizers/optimizer.h>

//
// The "Adaptive Moment" (Adam) Optimizer
// for more info, see : https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
//
struct Adam : Optimizer
{
    // hyper parameters
    float learning_rate, beta1, beta2;

    Adam();
    Adam(float _learning_rate, float _beta1, float _beta2);

    // tweak the parameters & zero thier gradients
    void update(Array<Parameter> &parameters);

    // save hyper parameters to a file
    void save(std::ofstream& file);

    // load hyper parameters from a file
    void load(std::ifstream& file);
};
