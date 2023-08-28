#pragma once

#include <cstdlib>

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
