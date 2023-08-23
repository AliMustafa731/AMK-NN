#pragma once

#include <cassert>
#include "data/array.h"
#include "data/geometry.h"
#include "common.h"

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

__forceinline void convolution(Matrix a, Matrix b, Matrix c, Shape stride)
{
    int w = _min((a.w - b.w) / stride.w + 1, c.w);
    int h = _min((a.h - b.h) / stride.h + 1, c.h);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float z = 0;

            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    z += b(_x, _y) * a(_x + x * stride.w, _y + y * stride.h);
                }
            }

            c(x, y) += z;
        }
    }
}

__forceinline void convolution_ex(Matrix a, Matrix b, Matrix c, Shape dialation)
{
    int w = _min(a.w - ((b.w - 1) * dialation.w + 1) + 1, c.w);
    int h = _min(a.h - ((b.h - 1) * dialation.h + 1) + 1, c.h);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float z = 0;

            for (int _y = 0; _y < b.h; _y++)
            {
                for (int _x = 0; _x < b.w; _x++)
                {
                    z += b(_x, _y) * a(x + _x * dialation.w, y + _y * dialation.h);
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
                    c(x * stride.w + _x, y * stride.h + _y) += a(x, y) * b(_x, _y);
                }
            }
        }
    }
}

__forceinline void multiply(float* data, int size, float value)
{
    for (int i = 0; i < size; i++)
    {
        data[i] *= value;
    }
}

__forceinline void copy_matrix(Matrix &dest, Matrix &src, Rect dest_rect, Rect src_rect)
{
    for (int y = src_rect.y; y < src_rect.h + src_rect.y; y++)
    {
        for (int x = src_rect.x; x < src_rect.w + src_rect.x; x++)
        {
            dest(x + dest_rect.x - src_rect.x, y + dest_rect.y - src_rect.y) = src(x, y);
        }
    }
}

__forceinline void copy_matrix(Matrix &dest, Matrix &src, Rect dest_rect, Rect src_rect, Shape stride)
{
    for (int y = src_rect.y; y < src_rect.h + src_rect.y; y++)
    {
        for (int x = src_rect.x; x < src_rect.w + src_rect.x; x++)
        {
            dest(x*(stride.w) + dest_rect.x - src_rect.x, y*(stride.h) + dest_rect.y - src_rect.y) = src(x, y);
        }
    }
}

__forceinline void copy(float *dest, float *src, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
}

__forceinline void copy(Array<float> dest, Array<float> src)
{
    AMK_ASSERT(dest.size() == src.size());
    for (int i = 0; i < dest.size(); i++)
    {
        dest[i] = src[i];
    }
}

__forceinline void copy_add(Array<float> dest, Array<float> src)
{
    AMK_ASSERT(dest.size() == src.size());
    for (int i = 0; i < dest.size(); i++)
    {
        dest[i] += src[i];
    }
}

__forceinline void fill_array(float *dest, float value, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = value;
    }
}

__forceinline void fill_array(Array<float> dest, float value)
{
    for (int i = 0; i < dest.size(); i++)
    {
        dest[i] = value;
    }
}

__forceinline void rotate_matrix(Matrix &dest, Matrix &src)
{
    AMK_ASSERT(dest.w == src.w && dest.h == src.h);
    int size = dest.w * dest.h;

    for (int i = 0; i < size; i++)
    {
        dest.data[i] = src.data[size - 1 - i];
    }
}

__forceinline int rand32()
{
    int a = rand();
    int b = rand();
    return a | b << 15;
}

__forceinline float random()
{
    return ((float)rand() / (float)RAND_MAX);
}

__forceinline float random(float a, float b)
{
    return a + (b - a)*random();
}

__forceinline void randomize(float* data, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = random();
    }
}

__forceinline void randomize(float* data, uint32_t size, float a, float b)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = random(a, b);
    }
}
