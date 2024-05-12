#pragma once

#include <optimizers/parameter.h>
#include <data/array.h>
#include <fstream>

struct Optimizer
{
    int type;

    Optimizer() {}
    virtual ~Optimizer() {}

    virtual void update(Array<Parameter> &parameters) = 0;
    virtual void save(std::ofstream& file) = 0;
    virtual void load(std::ifstream& file) = 0;
};
