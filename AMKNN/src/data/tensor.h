#pragma once

#include <cstring>
#include <cassert>
#include <fstream>
#include <cstdint>
#include "common.h"
#include "data/shape.h"

//----------------------------------------------------
//   Array, with Arithmetic functions & 2D accesors
//----------------------------------------------------
template<typename T> struct Tensor
{
    T *data;
    Shape shape;

    Tensor() : shape(0, 0, 0, 0), data(NULL) {};
    Tensor(int w, int h = 1, int d = 1, T* _data = NULL)
    {
        init({w, h, d, 1}, _data);
    }
    Tensor(Shape s, T* _data = NULL)
    {
        init(s, _data);
    }

    // arithmetic
    inline void add(Tensor<T>& rhs) { for (int i = 0; i < shape.size(); i++) data[i] += rhs.data[i]; };
    inline void sub(Tensor<T>& rhs) { for (int i = 0; i < shape.size(); i++) data[i] -= rhs.data[i]; };
    inline void mul(T factor)       { for (int i = 0; i < shape.size(); i++) data[i] *= factor; };

    inline void fill(T val) { for (int i = 0; i < shape.size(); i++) data[i] = val; }

    // accessors
    inline int size() { return shape.size(); }

    inline T  operator()(size_t x, size_t y) const { AMK_ASSERT(x < shape[0] && y < shape[1]);  return data[x + y * shape[0]]; }
    inline T& operator()(size_t x, size_t y)       { AMK_ASSERT(x < shape[0] && y < shape[1]);  return data[x + y * shape[1]]; }
    inline T  operator[](size_t i) const        { AMK_ASSERT(i < size());  return data[i]; }
    inline T& operator[](size_t i)              { AMK_ASSERT(i < size());  return data[i]; }

    // utilites
    void init(Shape s, T* _data = NULL)
    {
        shape = s;

        if (_data != NULL)
        {
            data = _data;
        }
        else
        {
            data = new T[shape.size()];
            std::memset(data, 0, size() * sizeof(T));
        }
    }

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
            shape = Shape(0, 0, 0, 0);
        }
    }

    void save(std::ofstream& file)
    {
        if (data != NULL)
        {
            file.write((char*)&shape, sizeof(Shape));
            file.write((char*)data, shape.size() * sizeof(T));
        }
    }

    void load(std::ifstream& file)
    {
        if (data != NULL)
        {
            file.read((char*)&shape, sizeof(Shape));
            file.read((char*)data, shape.size() * sizeof(T));
        }
    }
};
