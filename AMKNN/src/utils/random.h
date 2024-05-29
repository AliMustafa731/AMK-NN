#pragma once

#include <cstdlib>  // for rand()
#include <cstdint>  // for size_t

/// @return 32-bit random integer
__forceinline int rand32()
{
    int a = rand();
    int b = rand();
    return a | b << 15;
}

/// generate random 32-bit floating-point number in range [0.0, 1.0]
__forceinline float random()
{
    return ((float)rand() / (float)RAND_MAX);
}

/// generate random 32-bit floating-point number in range [a, b]
/// @param a : lower range
/// @param b : upper range
__forceinline float random(float a, float b)
{
    return a + (b - a)*random();
}

/// set random values for an array of 32-bit floating-point numbers in range [0.0, 1.0]
/// @param data : pointer to the array
/// @param size : number of elements in the array
__forceinline void randomize(float* data, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = random();
    }
}

/// set random values for an array of 32-bit floating-point numbers in range [a, b]
/// @param data : pointer to the array
/// @param size : number of elements in the array
/// @param a : lower range
/// @param b : upper range
__forceinline void randomize(float* data, size_t size, float a, float b)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = random(a, b);
    }
}
