#pragma once

#include <fstream>
#include <cstdint>
#include "shape.h"

//  Declaration
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
    Tensor<T> slice(Shape s, Shape offset);

    void save(std::ofstream& file);
    void load(std::ifstream& file);

    void copyFrom(Tensor<T> &src, Shape rect, Shape src_offset, Shape dest_offset);
};


// implementation
#ifdef TESNOR_IMPLEMENTATION

#include <common.h>

// constructors
template<typename T>
Tensor<T>::Tensor() : shape(0, 0, 0, 0), data(NULL) {}

template<typename T>
Tensor<T>::Tensor(int w, int h, int d, T* _data)
{
    init({ w, h, d, 1 }, _data);
}

template<typename T>
Tensor<T>::Tensor(Shape s, T* _data)
{
    init(s, _data);
}

// arithmetic
template<typename T>
void Tensor<T>::add(Tensor<T>& rhs) { for (int i = 0; i < shape.size(); i++) data[i] += rhs.data[i]; }

template<typename T>
void Tensor<T>::sub(Tensor<T>& rhs) { for (int i = 0; i < shape.size(); i++) data[i] -= rhs.data[i]; }

template<typename T>
void Tensor<T>::mul(T factor) { for (int i = 0; i < shape.size(); i++) data[i] *= factor; }

template<typename T>
void Tensor<T>::fill(T val) { for (int i = 0; i < shape.size(); i++) data[i] = val; }

// accessors
template<typename T>
int Tensor<T>::size() const { return shape.size(); }

template<typename T>
T  Tensor<T>::operator[](size_t i) const
{
    AMK_ASSERT(i < size());
    return data[i];
}

template<typename T>
T& Tensor<T>::operator[](size_t i)
{
    AMK_ASSERT(i < size());
    return data[i];
}

template<typename T>
T  Tensor<T>::operator()(size_t x, size_t y) const
{
    AMK_ASSERT(x < shape[0] && y < shape[1]);
    return data[x + y * dim_mul[0]];
}

template<typename T>
T& Tensor<T>::operator()(size_t x, size_t y)
{
    AMK_ASSERT(x < shape[0] && y < shape[1]);
    return data[x + y * dim_mul[0]];
}

template<typename T>
inline T Tensor<T>::operator()(size_t x, size_t y, size_t w) const
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2]);
    return data[x + y * dim_mul[0] + w * dim_mul[1]];
}

template<typename T>
inline T & Tensor<T>::operator()(size_t x, size_t y, size_t w)
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2]);
    return data[x + y * dim_mul[0] + w * dim_mul[1]];
}

template<typename T>
inline T Tensor<T>::operator()(size_t x, size_t y, size_t w, size_t h) const
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2] && h < shape[3]);
    return data[x + y * dim_mul[0] + w * dim_mul[1] + h * dim_mul[2]];
}

template<typename T>
inline T & Tensor<T>::operator()(size_t x, size_t y, size_t w, size_t h)
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2] && h < shape[3]);
    return data[x + y * dim_mul[0] + w * dim_mul[1] + h * dim_mul[2]];
}

// utilites
template<typename T>
void Tensor<T>::init(Shape s, T* _data)
{
    reshape(s);

    if (_data != NULL)
    {
        data = _data;
    }
    else
    {
        data = new T[shape.size()];
        std::memset(data, 0, size() * sizeof(T));
    }
}

template<typename T>
void Tensor<T>::release()
{
    if (data != NULL)
    {
        delete[] data;
        data = NULL;
        shape = Shape(0, 0, 0, 0);
    }
}

template<typename T>
void Tensor<T>::save(std::ofstream& file)
{
    if (data != NULL)
    {
        file.write((char*)&shape, sizeof(Shape));
        file.write((char*)data, shape.size() * sizeof(T));
    }
}

template<typename T>
void Tensor<T>::load(std::ifstream& file)
{
    if (data != NULL)
    {
        file.read((char*)&shape, sizeof(Shape));
        file.read((char*)data, shape.size() * sizeof(T));
    }
}

template<typename T>
void Tensor<T>::reshape(Shape s)
{
    shape = s;

    dim_mul[0] = shape[0];
    dim_mul[1] = shape[0] * shape[1];
    dim_mul[2] = shape[0] * shape[1] * shape[2];
}

template<typename T>
Tensor<T> Tensor<T>::slice(Shape s, Shape offset)
{
    int x = offset[0];
    int y = offset[1];
    int w = offset[2];
    int h = offset[3];
    return Tensor<T>(s, &data[x + y * dim_mul[0] + w * dim_mul[1] + h * dim_mul[2]]);
}

template<typename T>
void Tensor<T>::copyFrom(Tensor<T> &src, Shape rect, Shape src_offset, Shape dest_offset)
{
    AMK_ASSERT(
        rect[0] + src_offset[0] <= src.shape[0] &&
        rect[1] + src_offset[1] <= src.shape[1] &&
        rect[2] + src_offset[2] <= src.shape[2] &&
        rect[3] + src_offset[3] <= src.shape[3] &&

        rect[0] + dest_offset[0] <= this->shape[0] &&
        rect[1] + dest_offset[1] <= this->shape[1] &&
        rect[2] + dest_offset[2] <= this->shape[2] &&
        rect[3] + dest_offset[3] <= this->shape[3]
    );

    for (int w = 0; w < rect[3]; w++)
     for (int z = 0; z < rect[2]; z++)
      for (int y = 0; y < rect[1]; y++)
       for (int x = 0; x < rect[0]; x++)
       {
           (*this)(x + dest_offset[0], y + dest_offset[1], z + dest_offset[2], w + dest_offset[3])
           = src(x + src_offset[0], y + src_offset[1], z + src_offset[2], w + src_offset[3]);
       }
}

#endif  // TESNOR_IMPLEMENTATION
