#pragma once

#include <cassert>
#include "common.h"

//----------------------------------------------------
//   Static Array, with 2D meta-info & accesors
//----------------------------------------------------
template<typename T> struct Buffer
{
    T *data;
    int w, h, size;

    Buffer() { w = 0;  h = 0;  size = 0;  data = NULL; }
    Buffer(int _w, int _h) { init(_w, _h); }
    Buffer(int _w, int _h, T* _data) { init(_w, _h, _data); }

    inline T operator()(int x, int y) const { AMK_ASSERT(x < w && y < h);  return data[x + y * w]; }
    inline T &operator()(int x, int y) { AMK_ASSERT(x < w && y < h);  return data[x + y * w]; }
    inline T operator[](int i) const { AMK_ASSERT(i < size);  return data[i]; }
    inline T &operator[](int i) { AMK_ASSERT(i < size);  return data[i]; }

    void init(int _w, int _h)
    {
        w = _w;
        h = _h;
        size = w * h;
        data = new T[w * h];

        // initialize the memory to zero
        unsigned char* p = (unsigned char*)data;
        for (int i = 0; i < size * sizeof(T); i++) { p[i] = 0; }
    }

    void init(int _w, int _h, T* _data)
    {
        w = _w;
        h = _h;
        size = w * h;
        data = _data;
    }

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
            size = 0;
            w = 0;
            h = 0;
        }
    }
};
