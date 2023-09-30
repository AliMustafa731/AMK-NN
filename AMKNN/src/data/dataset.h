#pragma once

#include <cassert>
#include <common.h>
#include <utils/geometry.h>
#include <data/array.h>
#include <data/tensor.h>


struct DataSet
{
public:
    int sample_size, samples_num;
    Shape shape;
    Tensor<float> data;

    DataSet() {}

    void init(Shape _shape, int _samples_num);
    void release();

    // accessors
    inline Tensor<float> operator[](int i) const
    {
        AMK_ASSERT(i < samples_num);
        return Tensor<float>(shape.w, shape.h, shape.d, &data.data[i * sample_size]);
    }
    inline Tensor<float>& operator[](int i)
    {
        AMK_ASSERT(i < samples_num);
        ptr = Tensor<float>(shape.w, shape.h, shape.d, &data.data[i * sample_size]);
        return ptr;
    }

private:
    Tensor<float> ptr;  // used to lookup samples
};
