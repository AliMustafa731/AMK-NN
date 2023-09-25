#pragma once

#include <cstring>
#include <cassert>
#include "common.h"
#include "utils/geometry.h"

//----------------------------------------------------
//   Array, with Arithmetic functions 2D accesors
//----------------------------------------------------
template<typename T> struct Tensor
{
    T *data;
    Shape s;

    Tensor() { s = Shape(0, 0, 0); data = NULL; }
    Tensor(int w, int h = 1, int d = 1, T* _data = NULL) { init(w, h, d, _data); }

    inline int size() const { return s.w * s.h * s.d; }

    // arithmetic
    inline void add(Tensor<T>& rhs) { for (int i = 0; i < s.size(); i++) data[i] += rhs.data[i]; };
    inline void sub(Tensor<T>& rhs) { for (int i = 0; i < s.size(); i++) data[i] -= rhs.data[i]; };
    inline void mul(T factor)       { for (int i = 0; i < s.size(); i++) data[i] *= factor; };

    inline void fill(T val) { for (int i = 0; i < s.size(); i++) data[i] = val; }

    // accessors
    inline T  operator()(int x, int y) const { AMK_ASSERT(x < s.w && y < s.h);  return data[x + y * s.w]; }
    inline T& operator()(int x, int y)       { AMK_ASSERT(x < s.w && y < s.h);  return data[x + y * s.w]; }
    inline T  operator[](int i) const        { AMK_ASSERT(i < size());  return data[i]; }
    inline T& operator[](int i)              { AMK_ASSERT(i < size());  return data[i]; }

    void init(int w, int h = 1, int d = 1, T* _data = NULL)
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

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
            s = Shape(0, 0, 0);
        }
    }
};
