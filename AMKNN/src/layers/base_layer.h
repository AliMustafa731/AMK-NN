#pragma once

#include <data/array.h>
#include <data/tensor.h>
#include <optimizers/parameter.h>
#include <utils/geometry.h>
#include <fstream>

struct NeuralLayer
{
    Tensor<float> X, dX, Y, dY;
    Array<Parameter> parameters;
    int in_size, out_size, type;
    Shape in_shape, out_shape;
    bool trainable;

    NeuralLayer() {}

    void allocate(int _in_size, int _out_size);
    void deallocate();
    virtual void init(Shape _in_shape) = 0; // implemented by derrived classes
    virtual void release();

    void setTrainable(bool state);

    virtual void save(std::ofstream& file);
    virtual void load(std::ifstream& file);
    virtual Tensor<float>& forward(Tensor<float>& input) = 0;
    virtual Tensor<float>& backward(Tensor<float>& output_grad) = 0;

    static NeuralLayer* loadFromFile(std::ifstream& file);
};
