#pragma once

#include <cstring>
#include <cassert>
#include <common.h>

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
        w = _w; h = _h;  data = _data;
    }
    Matrix(int _w, int _h)
    {
        init(_w, _h);
    }

    inline float  operator()(int x, int y) const { AMK_ASSERT(x < w && y < h); return data[x + y * w]; }
    inline float& operator()(int x, int y)       { AMK_ASSERT(x < w && y < h); return data[x + y * w]; }

    void init(int _w, int _h)
    {
        w = _w;
        h = _h;
        data = new float[w * h];
        std::memset(data, 0, w * h * sizeof(float));
    }

    inline Shape get_shape() { return { w, h, 1 }; }

    inline Rect get_rect() { return { 0, 0, w, h }; }

    static void copy(Matrix &dest, Matrix &src, Rect dest_rect, Rect src_rect)
    {
        for (int y = src_rect.y; y < src_rect.h + src_rect.y; y++)
        {
            for (int x = src_rect.x; x < src_rect.w + src_rect.x; x++)
            {
                dest(x + dest_rect.x - src_rect.x, y + dest_rect.y - src_rect.y) = src(x, y);
            }
        }
    }

    static void copy(Matrix &dest, Matrix &src, Rect dest_rect, Rect src_rect, Shape stride)
    {
        for (int y = src_rect.y; y < src_rect.h + src_rect.y; y++)
        {
            for (int x = src_rect.x; x < src_rect.w + src_rect.x; x++)
            {
                dest(x*(stride.w) + dest_rect.x - src_rect.x, y*(stride.h) + dest_rect.y - src_rect.y) = src(x, y);
            }
        }
    }
};
