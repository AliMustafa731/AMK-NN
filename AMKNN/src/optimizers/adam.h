#pragma once

#include <optimizers/optimizer.h>

struct Adam : Optimizer
{
    float learning_rate, beta1, beta2;

    Adam();
    Adam(float _learning_rate, float _beta1, float _beta2);

    void update(Array<Parameter> &parameters);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
