#pragma once

#include <data/tensor.h>

struct Parameter
{
    Tensor<float> values, gradients, velocities, squared_gradients;
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
