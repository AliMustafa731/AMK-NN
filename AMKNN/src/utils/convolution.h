#pragma once

#include <utils/geometry.h>
#include <algorithm>  // for std::min

/// 2D Convolution
/// @param a : Input matrix
/// @param b : kernel matrix
/// @param c : Buffer to store the result
__forceinline void convolution(Matrix a, Matrix b, Matrix c)
{
    int w = std::min(a.w - b.w + 1, c.w);
    int h = std::min(a.h - b.h + 1, c.h);

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

/// Strided 2D Convolution
/// @param a : Input matrix
/// @param b : kernel matrix
/// @param c : Buffer to store the result
/// @param stride : step-size of sliding the Kernel over the Input
__forceinline void convolution_stride(Matrix a, Matrix b, Matrix c, Shape stride)
{
    int w = std::min((a.w - b.w) / stride[0] + 1, c.w);
    int h = std::min((a.h - b.h) / stride[1] + 1, c.h);

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

/// Dialated 2D Convolution
/// @param a : Input matrix
/// @param b : kernel matrix
/// @param c : Buffer to store the result
/// @param dialation : spacing between the elemnts of the Kernel when sliding over the Input
__forceinline void convolution_dialate(Matrix a, Matrix b, Matrix c, Shape dialation)
{
    int w = std::min(a.w - ((b.w - 1) * dialation[0] + 1) + 1, c.w);
    int h = std::min(a.h - ((b.h - 1) * dialation[1] + 1) + 1, c.h);

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

/// Transposed 2D Convolution
/// @param a : Input matrix
/// @param b : kernel matrix
/// @param c : Buffer to store the result
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

/// Strided Transposed 2D Convolution
/// @param a : Input matrix
/// @param b : kernel matrix
/// @param c : Buffer to store the result
/// @param stride : step-size of sliding the Kernel over the Input
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
