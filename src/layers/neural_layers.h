#pragma once

#include "optimizers.h"
#include "data/array.h"
#include "data/geometry.h"
#include "common.h"
#include <fstream>

#define FULL_LAYER 0
#define CONV_LAYER 1
#define CONV_TRANSPOSE_LAYER 2
#define MAX_POOL_LAYER 3
#define AVG_POOL_LAYER 4
#define RElU_LAYER 5
#define SIGMOID_LAYER 6
#define TANH_LAYER 7
#define RELU_LEAK_LAYER 8
#define DROPOUT_LAYER 9
#define SINE_LAYER 10

//----------------------------------------------
//  Base Layer Class
//----------------------------------------------
struct NeuralLayer
{
    Array<float> X, dX, Y, dY;
    int in_size, out_size, type;
    Shape in_shape, out_shape;
    std::vector<Parameter> parameters;
    bool trainable;

    NeuralLayer(){}

    void allocate(int _in_size, int _out_size);
    void deallocate();
    void save_parameters(std::ofstream& file);
    void load_parameters(std::ifstream& file);
    void setTrainable(bool state);

    virtual void init(Shape _in_shape) = 0;
    virtual float* forward(float* input) = 0;
    virtual float* backward(float* d_output) = 0;
    virtual void save(std::ofstream& file) = 0;
    virtual void load(std::ifstream& file) = 0;
};

//----------------------------------------------
//  Fully Connected Layer
//----------------------------------------------
struct FullLayer : NeuralLayer
{
    Array<float> W, dW, B, dB;
    float weight_decay;

    FullLayer()
    {
        type = FULL_LAYER;
    }
    FullLayer(int _size, float _weight_decay = 0, Shape _out_shape = Shape(0, 0, 0))
    {
        trainable = true;
        weight_decay = _weight_decay;
        type = FULL_LAYER;
        out_size = _size;

        if (_out_shape.size() == 0)
        {
            out_shape = { out_size, 1, 1 };
        }
        else
        {
            out_shape = _out_shape;
        }
    }

    void init(Shape _in_shape);
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

//----------------------------------------------
//  Convolutional Layer
//----------------------------------------------
struct ConvLayer : NeuralLayer
{
    Array<float> K, dK, B, dB;
    Matrix _X, _X_padd, _dX, _dX_padd, _Y, _dY, _K, _dK;
    Shape kernel, padd, stride;
    float weight_decay;

    ConvLayer()
    {
        type = CONV_LAYER;
    }
    ConvLayer(Shape _kernel, Shape _stride = Shape(1, 1, 0), Shape _padd = Shape(0, 0, 0), float _weight_decay = 0)
    {
        trainable = true;
        type = CONV_LAYER;
        kernel = _kernel;
        stride = _stride;
        padd = _padd;
        weight_decay = _weight_decay;
    }

    void init(Shape _in_shape);
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

//----------------------------------------------
//  Transposed Convolutional Layer
//----------------------------------------------
struct ConvTLayer : NeuralLayer
{
    Array<float> K, dK, B, dB;
    Matrix _X, _dX, _Y, _Y_padd, _dY, _K, _dK;
    Shape kernel, padd, stride;
    float weight_decay;

    ConvTLayer()
    {
        type = CONV_TRANSPOSE_LAYER;
    }
    ConvTLayer(Shape _kernel, Shape _stride = Shape(1, 1, 0), Shape _padd = Shape(0, 0, 0), float _weight_decay = 0)
    {
        trainable = true;
        type = CONV_TRANSPOSE_LAYER;
        kernel = _kernel;
        stride = _stride;
        padd = _padd;
        weight_decay = _weight_decay;
    }

    void init(Shape _in_shape);
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

//----------------------------------------------
//  Pooling Layers
//----------------------------------------------
struct MaxPoolLayer : NeuralLayer
{
    Shape window, stride;
    int *max_indices;

    MaxPoolLayer()
    {
        type = MAX_POOL_LAYER;
    }
    MaxPoolLayer(Shape _window, Shape _stride)
    {
        type = MAX_POOL_LAYER;
        window = _window;
        stride = _stride;
    }

    void init(Shape _in_shape);
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};

struct AvgPoolLayer : NeuralLayer
{
    Shape window, stride;

    AvgPoolLayer()
    {
        type = AVG_POOL_LAYER;
    }
    AvgPoolLayer(Shape _window, Shape _stride)
    {
        type = AVG_POOL_LAYER;
        window = _window;
        stride = _stride;
    }

    void init(Shape _in_shape);
    float* forward(float* input);
    float* backward(float* d_output);
    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
