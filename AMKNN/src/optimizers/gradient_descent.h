#pragma once

#include "optimizers/optimizer.h"

struct GradientDescent : Optimizer
{
    float learning_rate, momentum;

    GradientDescent();
    GradientDescent(float _learning_rate, float _momentum);

    void update(List<Parameter*> &parameters);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
