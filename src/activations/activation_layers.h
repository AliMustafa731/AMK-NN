#pragma once

#include "common.h"
#include "layers/neural_layers.h"


//----------------------------------------------
//  Layers that perform non-linear functions
//----------------------------------------------
struct RelULayer : NeuralLayer
{
    RelULayer();

    void init(Shape _in_shape);
    void release(){}
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct RelULeakLayer : NeuralLayer
{
    float alpha;

    RelULeakLayer(float _alpha);
    RelULeakLayer();

    void init(Shape _in_shape);
    void release(){}
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

struct SigmoidLayer : NeuralLayer
{
    SigmoidLayer();

    void init(Shape _in_shape);
    void release(){}
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct TanhLayer : NeuralLayer
{
    TanhLayer();

    void init(Shape _in_shape);
    void release(){}
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct SineLayer : NeuralLayer
{
    SineLayer();

    void init(Shape _in_shape);
    void release(){}
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file) {}
    void load(std::ifstream& file) {}
};

struct DropoutLayer : NeuralLayer
{
    Array<float> mask;  // contains values that can have either (0) or (1)
    float P;            // the probability of nodes being kept working

    DropoutLayer(float _P);
    DropoutLayer();

    void init(Shape _in_shape);
    void release();
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
