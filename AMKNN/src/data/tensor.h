#pragma once

#include <fstream>
#include <cstdint>
#include "common.h"
#include "data/shape.h"

//----------------------------------------------------
//   Array, with Arithmetic functions & 2D accesors
//----------------------------------------------------
template<typename T> struct Tensor
{
    T *data;
    Shape shape;
    size_t dim_mul[3];

    Tensor();
    Tensor(int w, int h = 1, int d = 1, T* _data = NULL);
    Tensor(Shape s, T* _data = NULL);

    // arithmetic
    inline void add(Tensor<T>& rhs);
    inline void sub(Tensor<T>& rhs);
    inline void mul(T factor);

    inline void fill(T val);

    // accessors
    inline int size() const;

    inline T  operator[](size_t i) const;
    inline T& operator[](size_t i);

    inline T  operator()(size_t x, size_t y) const;
    inline T& operator()(size_t x, size_t y);

    inline T  operator()(size_t x, size_t y, size_t w) const;
    inline T& operator()(size_t x, size_t y, size_t w);

    inline T  operator()(size_t x, size_t y, size_t w, size_t h) const;
    inline T& operator()(size_t x, size_t y, size_t w, size_t h);

    // utilites
    void init(Shape s, T* _data = NULL);
    void release();

    void reshape(Shape s);

    void save(std::ofstream& file);
    void load(std::ifstream& file);
};
