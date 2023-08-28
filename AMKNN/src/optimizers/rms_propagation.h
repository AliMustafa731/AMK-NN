#pragma once

#include "optimizers/optimizer.h"

struct RMSPropagation : Optimizer
{
    float learning_rate, beta;

    RMSPropagation();
    RMSPropagation(float _learning_rate, float _beta);

    void update(List<Parameter*> &parameters);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
