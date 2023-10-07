#pragma once

#include <data/tensor.h>
#include <fstream>
#include <cstdint>

struct Parameter
{
    Tensor<float> values, gradients, velocities, squared_gradients;
    size_t _size;
    float decay_rate;
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

    inline size_t size() { return _size; }

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

    void release()
    {
        values.release();
        gradients.release();
        velocities.release();
        squared_gradients.release();
    }

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
