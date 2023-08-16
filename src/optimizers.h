#pragma once

#include <vector>
#include <fstream>
#include "utils/data.h"
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

struct NeuralNetwork;

struct Optimizer
{
    int type;

    Optimizer() {}
    virtual ~Optimizer() {}

    virtual void update(std::vector<Parameter*> &parameters) = 0;
    virtual void save(std::ofstream& file) = 0;
    virtual void load(std::ifstream& file) = 0;
};

struct Adam : Optimizer
{
    float learning_rate, beta1, beta2;

    Adam()
    {
        type = ADAM;
    }
    Adam(float _learning_rate, float _beta1, float _beta2)
    {
        type = ADAM;
        learning_rate = _learning_rate;
        beta1 = _beta1;
        beta2 = _beta2;
    }

    void update(std::vector<Parameter*> &parameters);

    void save(std::ofstream& file)
    {
        file.write((char*)&learning_rate, sizeof(float));
        file.write((char*)&beta1, sizeof(float));
        file.write((char*)&beta2, sizeof(float));
    }
    void load(std::ifstream& file)
    {
        file.read((char*)&learning_rate, sizeof(float));
        file.read((char*)&beta1, sizeof(float));
        file.read((char*)&beta2, sizeof(float));
    }
};

struct GradientDescent : Optimizer
{
    float learning_rate, momentum;

    GradientDescent()
    {
        type = GRADIENT_DESCENT;
    }
    GradientDescent(float _learning_rate, float _momentum)
    {
        type = GRADIENT_DESCENT;
        learning_rate = _learning_rate;
        momentum = _momentum;
    }

    void update(std::vector<Parameter*> &parameters);

    void save(std::ofstream& file)
    {
        file.write((char*)&learning_rate, sizeof(float));
        file.write((char*)&momentum, sizeof(float));
    }
    void load(std::ifstream& file)
    {
        file.read((char*)&learning_rate, sizeof(float));
        file.read((char*)&momentum, sizeof(float));
    }
};

struct RMSPropagation : Optimizer
{
    float learning_rate, beta;

    RMSPropagation()
    {
        type = RMS_PROPAGATION;
    }
    RMSPropagation(float _learning_rate, float _beta)
    {
        type = RMS_PROPAGATION;
        learning_rate = _learning_rate;
        beta = _beta;
    }

    void update(std::vector<Parameter*> &parameters);

    void save(std::ofstream& file)
    {
        file.write((char*)&learning_rate, sizeof(float));
        file.write((char*)&beta, sizeof(float));
    }
    void load(std::ifstream& file)
    {
        file.read((char*)&learning_rate, sizeof(float));
        file.read((char*)&beta, sizeof(float));
    }
};
