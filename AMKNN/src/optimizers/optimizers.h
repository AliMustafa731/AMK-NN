#pragma once

#include <fstream>
#include "data/list.h"
#include "data/array.h"
#include "common.h"

#define GRADIENT_DESCENT 0
#define RMS_PROPAGATION 1
#define ADAM 2

struct Parameter
{
    Array<float> values, gradients, velocities, squared_gradients;
    int size;
    float decay_rate;
    bool is_trainable;

    Parameter(int _size, float _decay_rate = 0)
    {
        is_trainable = true;
        size = _size;
        decay_rate = _decay_rate;
        values.init(size);
        gradients.init(size);
        velocities.init(size);
        squared_gradients.init(size);
    }
    Parameter() { is_trainable = true; }

    void release()
    {
        values.release();
        gradients.release();
        velocities.release();
        squared_gradients.release();
    }
};

struct Optimizer
{
    int type;

    Optimizer() {}
    virtual ~Optimizer() {}

    virtual void update(List<Parameter*> &parameters) = 0;
    virtual void save(std::ofstream& file) = 0;
    virtual void load(std::ifstream& file) = 0;
};

struct Adam : Optimizer
{
    float learning_rate, beta1, beta2;

    Adam();
    Adam(float _learning_rate, float _beta1, float _beta2);

    void update(List<Parameter*> &parameters);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

struct GradientDescent : Optimizer
{
    float learning_rate, momentum;

    GradientDescent();
    GradientDescent(float _learning_rate, float _momentum);

    void update(List<Parameter*> &parameters);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

struct RMSPropagation : Optimizer
{
    float learning_rate, beta;

    RMSPropagation();
    RMSPropagation(float _learning_rate, float _beta);

    void update(List<Parameter*> &parameters);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
