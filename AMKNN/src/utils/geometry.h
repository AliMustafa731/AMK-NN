#pragma once

#include <string>
#include <cassert>
#include "common.h"

struct Rect
{
    int x, y, w, h;

    Rect() {}
    Rect(int _x, int _y, int _w, int _h)
    {
		x = _x;  y = _y;  w = _w; h = _h;
    }
};

struct Shape
{
    int w, h, d;

    inline int size() { return w * h * d; }

    Shape() {}
    Shape(int _w, int _h, int _d)
    {
        w = _w;  h = _h;  d = _d;
    }
};

struct Matrix
{
    float *data;
    int w, h;

    Matrix() {}
    Matrix(int _w, int _h, float* _data)
    {
        w = _w;
        h = _h;
        data = _data;
    }
    Matrix(Shape s, float* _data)
    {
        w = s.w;
        h = s.h;
        data = _data;
    }
    Matrix(int _w, int _h) { init(_w, _h); }
    Matrix(Shape s) { init(s.w, s.h); }

    inline float operator()(int x, int y) const { AMK_ASSERT(x < w && y < h); return data[x + y * w]; }
    inline float &operator()(int x, int y)      { AMK_ASSERT(x < w && y < h); return data[x + y * w]; }

    void init(int _w, int _h)
    {
        w = _w;
        h = _h;
        data = new float[w * h];
        fill(0);
    }

    inline Shape get_shape() { return { w, h, 1 }; }

    inline Rect get_rect() { return { 0, 0, w, h }; }

    void fill(float x)
    {
        for (int i = 0; i < w*h; i++)
        {
            data[i] = x;
        }
    }
};
