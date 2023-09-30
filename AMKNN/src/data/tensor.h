#pragma once

#include <cstring>
#include <cassert>
#include <fstream>
#include <cstdint>
#include "common.h"
#include "utils/geometry.h"

//----------------------------------------------------
//   Array, with Arithmetic functions & 2D accesors
//----------------------------------------------------
template<typename T> struct Tensor
{
    T *data;
    Shape s;

    Tensor()
    {
        s = Shape(0, 0, 0); data = NULL;
    }
    Tensor(size_t w, size_t h = 1, size_t d = 1, T* _data = NULL)
    {
        init(w, h, d, _data);
    }
    Tensor(Shape _s, T* _data = NULL)
    {
        init(_s.w, _s.h, _s.d, _data);
    }

    // arithmetic
    inline void add(Tensor<T>& rhs) { for (int i = 0; i < s.size(); i++) data[i] += rhs.data[i]; };
    inline void sub(Tensor<T>& rhs) { for (int i = 0; i < s.size(); i++) data[i] -= rhs.data[i]; };
    inline void mul(T factor)       { for (int i = 0; i < s.size(); i++) data[i] *= factor; };

    inline void fill(T val) { for (int i = 0; i < s.size(); i++) data[i] = val; }

    // accessors
    inline size_t size() const { return s.w * s.h * s.d; }

    inline T  operator()(size_t x, size_t y) const { AMK_ASSERT(x < s.w && y < s.h);  return data[x + y * s.w]; }
    inline T& operator()(size_t x, size_t y)       { AMK_ASSERT(x < s.w && y < s.h);  return data[x + y * s.w]; }
    inline T  operator[](size_t i) const        { AMK_ASSERT(i < size());  return data[i]; }
    inline T& operator[](size_t i)              { AMK_ASSERT(i < size());  return data[i]; }

    // utilites
    void init(size_t w, size_t h = 1, size_t d = 1, T* _data = NULL)
    {
        s.w = w;
        s.h = h;
        s.d = d;

        if (_data != NULL)
        {
            data = _data;
        }
        else
        {
            data = new T[s.size()];
            std::memset(data, 0, size() * sizeof(T));
        }
    }

    void init(Shape _s, T* _data = NULL)
    {
        init(_s.w, _s.h, _s.d, _data);
    }

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
            s = Shape(0, 0, 0);
        }
    }

    void save(std::ofstream& file)
    {
        if (data != NULL)
        {
            file.write((char*)&s, sizeof(Shape));
            file.write((char*)data, s.size() * sizeof(T));
        }
    }

    void load(std::ifstream& file)
    {
        if (data != NULL)
        {
            file.read((char*)&s, sizeof(Shape));
            file.read((char*)data, s.size() * sizeof(T));
        }
    }
};
