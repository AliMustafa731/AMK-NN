#pragma once

#include <cassert>
#include "common.h"
#include "data/geometry.h"
#include "data/array.h"


struct DataSet
{
    int sample_size, samples_num;
    Shape shape;
    Array<float> data;
    Array<float*> ptr; // pointers to smaples are stored here, making it easy to shuffle the dataset

    DataSet() {}
    ~DataSet() { release(); }

    void init(Shape _shape, int _samples_num);
    void release();

    inline float* operator[](int i) const
    {
        AMK_ASSERT(i < ptr.size);
        return ptr[i];
    }
    inline float* &operator[](int i)
    {
        AMK_ASSERT(i < ptr.size);
        return ptr[i];
    }
};

template<typename T> __forceinline void make_ptr_list(Array<T*> &dest, Array<T> &src, int element_size)
{
    int elements_num = src.size / element_size;
    dest.init(elements_num);

    for (int i = 0; i < elements_num; i++)
    {
        dest[i] = &src[i * element_size];
    }
}

