#pragma once

#include <cstring>
#include <cstdint>
#include <cassert>
#include <common.h>
#include "./data/shape.h"

struct Rect
{
    size_t x, y, w, h;

    Rect() {}
    Rect(size_t _x, size_t _y, size_t _w, size_t _h)
    {
        x = _x;  y = _y;  w = _w; h = _h;
    }
};

struct Matrix
{
    float *data;
    size_t w, h;

    Matrix() {}
    Matrix(size_t _w, size_t _h, float* _data)
    {
        w = _w; h = _h;  data = _data;
    }
    Matrix(size_t _w, size_t _h)
    {
        init(_w, _h);
    }

    inline float  operator()(size_t x, size_t y) const { AMK_ASSERT(x < w && y < h); return data[x + y * w]; }
    inline float& operator()(size_t x, size_t y)       { AMK_ASSERT(x < w && y < h); return data[x + y * w]; }

    void init(size_t _w, size_t _h)
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
                dest(x*(stride[0]) + dest_rect.x - src_rect.x, y*(stride[1]) + dest_rect.y - src_rect.y) = src(x, y);
            }
        }
    }
};
