#pragma once

#include <data/tensor.h>
#include <fstream>
#include <cstdint>

struct Parameter
{
    Tensor<float> values, gradients, velocities, squared_gradients;
    size_t size;
    float decay_rate;
    bool is_trainable;

    Parameter(size_t _size, float _decay_rate = 0)
    {
        init(_size, _decay_rate);
    }
    Parameter()
    {
        is_trainable = true;
        size = 0;
    }

    void init(size_t _size, float _decay_rate = 0)
    {
        is_trainable = true;
        size = _size;
        decay_rate = _decay_rate;
        values.init(size);
        gradients.init(size);
        velocities.init(size);
        squared_gradients.init(size);
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
        file.write((char*)&size, sizeof(size_t));
        file.write((char*)&decay_rate, sizeof(float));
        file.write((char*)&is_trainable, sizeof(bool));

        values.save(file);
        gradients.save(file);
        velocities.save(file);
        squared_gradients.save(file);
    }

    void load(std::ifstream& file)
    {
        file.read((char*)&size, sizeof(size_t));
        file.read((char*)&decay_rate, sizeof(float));
        file.read((char*)&is_trainable, sizeof(bool));

        values.load(file);
        gradients.load(file);
        velocities.load(file);
        squared_gradients.load(file);
    }
};
