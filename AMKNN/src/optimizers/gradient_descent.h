#pragma once

#include <optimizers/optimizer.h>

//
// The "Gradient Descent" Optimizer
// for more info, see : https://machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure/
//
struct GradientDescent : Optimizer
{
    // hyper parameters
    float learning_rate, momentum;

    GradientDescent();
    GradientDescent(float _learning_rate, float _momentum);

    // tweak the parameters & zero thier gradients
    void update(Array<Parameter> &parameters);

    // save hyper parameters to a file
    void save(std::ofstream& file);

    // load hyper parameters from a file
    void load(std::ifstream& file);
};
