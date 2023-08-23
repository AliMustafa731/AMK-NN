#pragma once

#include "data/array.h"

struct Color
{
    unsigned char b, g, r, a;

    Color() {}
    Color(unsigned char _r, unsigned char _g, unsigned char _b)
    {
        r = _r;  g = _g;  b = _b;  a = 0;
    }
};

struct Colorf
{
    float r, g, b;

    Colorf() {}
    Colorf(float _r, float _g, float _b)
    {
        r = _r;  g = _g;  b = _b;
    }
};

__forceinline void embed_one_channel_to_color(Color *dest, unsigned char* src, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i].r = src[i];
        dest[i].g = src[i];
        dest[i].b = src[i];
        dest[i].a = 0;
    }
}

__forceinline void embed_one_channel_to_color(Color* dest, float* src, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i].r = (unsigned char)src[i];
        dest[i].g = (unsigned char)src[i];
        dest[i].b = (unsigned char)src[i];
        dest[i].a = 0;
    }
}

__forceinline void rgb_to_float(Colorf *dest, Color *src, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i].r = (float)src[i].r;
        dest[i].g = (float)src[i].g;
        dest[i].b = (float)src[i].b;
    }
}

__forceinline void float_to_rgb(Color *dest, Colorf *src, int size)
{
    for (int i = 0; i < size; i++)
    {
        src[i].r = (unsigned char)dest[i].r;
        src[i].g = (unsigned char)dest[i].g;
        src[i].b = (unsigned char)dest[i].b;
    }
}

__forceinline void rgb_to_float_one_channel(float *dest, unsigned char *src, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = (float)src[i];
    }
}

__forceinline void float_to_rgb_one_channel(unsigned char *dest, float *src, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = (unsigned char)src[i];
    }
}

template<typename T> void arrange_channels_to_blocks(Array<T> &dest, Array<T> &src, int channels)
{
    for (int j = 0; j < channels; j++)
    {
        for (int i = 0; i < dest.size / channels; i++)
        {
            dest[i * channels] = src[i + j];
        }
    }
}

template<typename T> void arrange_channels_to_color(Array<T> &dest, Array<T> &src, int channels)
{
    for (int j = 0; j < channels; j++)
    {
        for (int i = 0; i < dest.size / channels; i++)
        {
            dest[i + j] = src[i * channels];
        }
    }
}
