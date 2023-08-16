#pragma once

#include "common.h"
#include "layers/neural_layers.h"


//----------------------------------------------
//  Layers that perform non-linear functions
//----------------------------------------------
struct RelULayer : NeuralLayer
{
    RelULayer()
    {
        type = RElU_LAYER;
    }

    void init(Shape _in_shape)
    {
        out_size = in_size;
        out_shape = in_shape;
    }
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct RelULeakLayer : NeuralLayer
{
    float alpha;

    RelULeakLayer(float _alpha)
    {
        alpha = _alpha;
        type = RELU_LEAK_LAYER;
    }
    RelULeakLayer()
    {
        type = RELU_LEAK_LAYER;
    }

    void init(Shape _in_shape)
    {
        out_size = in_size;
        out_shape = in_shape;
    }
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

struct SigmoidLayer : NeuralLayer
{
    SigmoidLayer()
    {
        type = SIGMOID_LAYER;
    }

    void init(Shape _in_shape)
    {
        out_size = in_size;
        out_shape = in_shape;
    }
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct TanhLayer : NeuralLayer
{
    TanhLayer()
    {
        type = TANH_LAYER;
    }

    void init(Shape _in_shape)
    {
        out_size = in_size;
        out_shape = in_shape;
    }
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct SineLayer : NeuralLayer
{
    SineLayer()
    {
        type = SINE_LAYER;
    }

    void init(Shape _in_shape)
    {
        out_size = in_size;
        out_shape = in_shape;
    }
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

//---------------------------------------------------------------------------------
//  Dropout Layer (used as regularizer, improves generalization of the network)
//---------------------------------------------------------------------------------
struct DropoutLayer : NeuralLayer
{
    Array<float> mask;  // contains values that can have either (0) or (1)
    float P;            // the probability of nodes being kept working

    DropoutLayer(float _P)
    {
        trainable = true;
        P = _P;
        type = DROPOUT_LAYER;
    }
    DropoutLayer()
    {
        trainable = true;
        type = DROPOUT_LAYER;
    }

    void init(Shape _in_shape);
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
