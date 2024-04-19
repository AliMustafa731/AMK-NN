#pragma once

#include <cassert>
#include <utils/geometry.h>
#include <common.h>

#define _max(a, b) (( (a) > (b) ) ? (a) : (b))
#define _min(a, b) (( (a) < (b) ) ? (a) : (b))


__forceinline void convolution(Matrix a, Matrix b, Matrix c)
{
    int w = _min(a.w - b.w + 1, c.w);
    int h = _min(a.h - b.h + 1, c.h);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float z = 0;

            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    z += b(_x, _y) * a(_x + x, _y + y);
                }
            }

            c(x, y) += z;
        }
    }
}

__forceinline void convolution_stride(Matrix a, Matrix b, Matrix c, Shape stride)
{
    int w = _min((a.w - b.w) / stride[0] + 1, c.w);
    int h = _min((a.h - b.h) / stride[1] + 1, c.h);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float z = 0;

            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    z += b(_x, _y) * a(_x + x * stride[0], _y + y * stride[1]);
                }
            }

            c(x, y) += z;
        }
    }
}

__forceinline void convolution_dialate(Matrix a, Matrix b, Matrix c, Shape dialation)
{
    int w = _min(a.w - ((b.w - 1) * dialation[0] + 1) + 1, c.w);
    int h = _min(a.h - ((b.h - 1) * dialation[1] + 1) + 1, c.h);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float z = 0;

            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    z += b(_x, _y) * a(x + _x * dialation[0], y + _y * dialation[1]);
                }
            }

            c(x, y) += z;
        }
    }
}

__forceinline void convolution_transpose(Matrix a, Matrix b, Matrix c)
{
    for (int y = 0; y < a.h; y++)
    {
        for (int x = 0; x < a.w; x++)
        {
            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    c(x + _x, y + _y) += a(x, y) * b(_x, _y);
                }
            }
        }
    }
}

__forceinline void convolution_transpose(Matrix a, Matrix b, Matrix c, Shape stride)
{
    for (int y = 0; y < a.h; y++)
    {
        for (int x = 0; x < a.w; x++)
        {
            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    c(x * stride[0] + _x, y * stride[1] + _y) += a(x, y) * b(_x, _y);
                }
            }
        }
    }
}
