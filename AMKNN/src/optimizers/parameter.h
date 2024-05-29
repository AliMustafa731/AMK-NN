#pragma once

#include <data/tensor.h>
#include <fstream>
#include <cstdint>

/*
 * Parameters Array (weights and biases)
 */
struct Parameter
{
    /// Tensor containing the values of the parameters
    Tensor<float> values;

    /// Tensor containing the gradients of the parameters
    Tensor<float> gradients;

    /// Tensor containing the moving average of gradients of the parameters
    Tensor<float> velocities;

    /// Tensor containing the moving average of the square of gradients of the parameters
    Tensor<float> squared_gradients;

    /// number of elements in this parameters array
    size_t _size;

    /// L2 regularization rate
    float decay_rate;

    /// flag indicating whether this parameter array can accumulate gradients
    bool is_trainable;

    Parameter(size_t _size, float _decay_rate = 0)
    {
        init(_size, _decay_rate);
    }
    Parameter()
    {
        is_trainable = true;
        _size = 0;
    }

    /// @return number of elements in this parameters array
    inline size_t size()
    {
        return _size;
    }

    /// initialize the Tensors with the given size
    /// @param _size : number of elements of the parameters array
    /// @param _decay_rate : L2 regularization rate
    void init(size_t _size, float _decay_rate = 0)
    {
        is_trainable = true;
        this->_size = _size;
        decay_rate = _decay_rate;
        values.init(_size);
        gradients.init(_size);
        velocities.init(_size);
        squared_gradients.init(_size);
    }

    /// release the memory allocated by the internal Tensors
    void release()
    {
        values.release();
        gradients.release();
        velocities.release();
        squared_gradients.release();
    }

    /// save the parameters array into a file
    /// @param file : handle to a previously openned file
    void save(std::ofstream& file)
    {
        file.write((char*)&_size, sizeof(size_t));
        file.write((char*)&decay_rate, sizeof(float));
        file.write((char*)&is_trainable, sizeof(bool));

        values.save(file);
        gradients.save(file);
        velocities.save(file);
        squared_gradients.save(file);
    }

    /// load the parameters array from a file
    /// @param file : handle to a previously openned file
    void load(std::ifstream& file)
    {
        file.read((char*)&_size, sizeof(size_t));
        file.read((char*)&decay_rate, sizeof(float));
        file.read((char*)&is_trainable, sizeof(bool));

        values.load(file);
        gradients.load(file);
        velocities.load(file);
        squared_gradients.load(file);
    }
};
