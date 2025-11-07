#pragma once

#include <fstream>
#include <cstring>
#include <cstdint>
#include "shape.h"

/*
 * Multi-Dimensional Array Data Structure
 */
template<typename T> class Tensor
{
public:
    /// pointer to the memory where elements is to be stored
    T *data;

    /// shape (dimensions) of the Tensor
    Shape shape;

    Tensor();
    Tensor(size_t w, size_t h = 1, size_t d = 1, T* _data = NULL);
    Tensor(Shape s, T* _data = NULL);

    /// arithmetic

    /// Add this Tensor with another (element by element)
    /// @param rhs : the other Tensor
    inline void add(Tensor<T>& rhs);

    /// Subtract this Tensor from another (element by element)
    /// @param rhs : the other Tensor
    inline void sub(Tensor<T>& rhs);

    /// Multiply all elements of this Tensor by a factor
    /// @param factor : factor of multiplication
    inline void mul(T factor);

    /// set the whole elemnts of the Tensor to "val"
    inline void fill(T val);

    /// @return The number of elements (size of it's shape)
    inline size_t size() const;

    /// setters and getters for Tensor access (read/write)
    inline T  operator[](size_t i) const;
    inline T& operator[](size_t i);

    inline T  operator()(size_t x, size_t y) const;
    inline T& operator()(size_t x, size_t y);

    inline T  operator()(size_t x, size_t y, size_t w) const;
    inline T& operator()(size_t x, size_t y, size_t w);

    inline T  operator()(size_t x, size_t y, size_t w, size_t h) const;
    inline T& operator()(size_t x, size_t y, size_t w, size_t h);

    /*
     * initialize the Tensor with the specified shape and data pointer 
     * 
     * @param s : shape of the Tensor
     * 
     * @param __data :
     * pointer to the data, if "NULL" (not passed),
     * new memory is allocated
     */
    void init(Shape s, T* _data = NULL);

    /// release the memory of the Tensor
    void release();

    /// change the shape of the Tensor
    void reshape(Shape s);

    /*
     * take a slice of this Tensor,
     *
     * @param s : the shape of the slice taken
     * 
     * @param offset : offset to the current memory allocated by this Tensor
     * 
     * @return
     * new Tensor with shape "s", pointing to the memory of this Tensor,
     * no new memory is allocated
     */
    Tensor<T> slice(Shape s, Shape offset);

    /// save Tensor into a file
    /// @param file : handle to a previously openned file
    void save(std::ofstream& file);

    /// load Tensor from a file
    /// @param file : handle to a previously openned file
    void load(std::ifstream& file);

    /*
     * copy part from another Tensor into this Tensor,
     *
     * @param src : Tensor to copy from
     * 
     * @param rect : shape of the part to be copied
     * 
     * @param src_offset : multi-dimensional offset into the "src" Tensor (to copy from)
     * 
     * @param dest_offset : multi-dimensional offset into this Tensor
     */
    void copyFrom(Tensor<T> &src, Shape rect, Shape src_offset, Shape dest_offset);

private:

    /// cached values of dimension's multiplicands,
    /// used to speed-up memory access.
    size_t dim_mul[3];
};


// implementation
#include <common.h>

// constructors
template<typename T> Tensor<T>::Tensor() : shape(0, 0, 0, 0), data(NULL)
{

}

template<typename T> Tensor<T>::Tensor(size_t w, size_t h, size_t d, T* _data)
{
    init({ w, h, d, 1 }, _data);
}

template<typename T> Tensor<T>::Tensor(Shape s, T* _data)
{
    init(s, _data);
}

// arithmetic
template<typename T> void Tensor<T>::add(Tensor<T>& rhs)
{
    for (int i = 0; i < shape.size(); i++)
    {
        data[i] += rhs.data[i];
    }
}

template<typename T> void Tensor<T>::sub(Tensor<T>& rhs)
{
    for (int i = 0; i < shape.size(); i++)
    {
        data[i] -= rhs.data[i];
    }
}

template<typename T> void Tensor<T>::mul(T factor)
{
    for (int i = 0; i < shape.size(); i++)
    {
        data[i] *= factor;
    }
}

// set the whole elemnts of the Tensor to "val"
template<typename T> void Tensor<T>::fill(T val)
{
    for (int i = 0; i < shape.size(); i++)
    {
        data[i] = val;
    }
}

// accessors
template<typename T> size_t Tensor<T>::size() const
{
    return shape.size();
}

//
// Brackets Overloaded Setters and Getters
//
template<typename T> T Tensor<T>::operator[](size_t i) const
{
    AMK_ASSERT(i < size());
    return data[i];
}

template<typename T> T& Tensor<T>::operator[](size_t i)
{
    AMK_ASSERT(i < size());
    return data[i];
}

template<typename T> T Tensor<T>::operator()(size_t x, size_t y) const
{
    AMK_ASSERT(x < shape[0] && y < shape[1]);
    return data[x + y * dim_mul[0]];
}

template<typename T> T& Tensor<T>::operator()(size_t x, size_t y)
{
    AMK_ASSERT(x < shape[0] && y < shape[1]);
    return data[x + y * dim_mul[0]];
}

template<typename T> inline T Tensor<T>::operator()(size_t x, size_t y, size_t w) const
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2]);
    return data[x + y * dim_mul[0] + w * dim_mul[1]];
}

template<typename T> inline T & Tensor<T>::operator()(size_t x, size_t y, size_t w)
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2]);
    return data[x + y * dim_mul[0] + w * dim_mul[1]];
}

template<typename T> inline T Tensor<T>::operator()(size_t x, size_t y, size_t w, size_t h) const
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2] && h < shape[3]);
    return data[x + y * dim_mul[0] + w * dim_mul[1] + h * dim_mul[2]];
}

template<typename T> inline T & Tensor<T>::operator()(size_t x, size_t y, size_t w, size_t h)
{
    AMK_ASSERT(x < shape[0] && y < shape[1] && w < shape[2] && h < shape[3]);
    return data[x + y * dim_mul[0] + w * dim_mul[1] + h * dim_mul[2]];
}

//
// initialize Tensor, with optional "data" pointer parameter
// memory is allocated only if "data" is NULL (e.g. not passed as argument).
//
template<typename T> void Tensor<T>::init(Shape s, T* _data)
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

// release the memory allocated by the Tensor
template<typename T> void Tensor<T>::release()
{
    if (data != NULL)
    {
        delete[] data;
        data = NULL;
        shape = Shape(0, 0, 0, 0);
    }
}

// save Tensor into a file
template<typename T> void Tensor<T>::save(std::ofstream& file)
{
    if (data != NULL)
    {
        file.write((char*)&shape, sizeof(Shape));
        file.write((char*)data, shape.size() * sizeof(T));
    }
}

// load Tensor from a file
template<typename T> void Tensor<T>::load(std::ifstream& file)
{
    if (data != NULL)
    {
        file.read((char*)&shape, sizeof(Shape));
        file.read((char*)data, shape.size() * sizeof(T));
    }
}

//
// change the shape of the Tensor,
// also, re-calculate the dimension's multiplicands
//
template<typename T> void Tensor<T>::reshape(Shape s)
{
    shape = s;

    dim_mul[0] = shape[0];
    dim_mul[1] = shape[0] * shape[1];
    dim_mul[2] = shape[0] * shape[1] * shape[2];
}

//
// take a slice of this Tensor,
// with a shape (s) and offset into the current memory (offset).
// *no new memory is allocated.
//
template<typename T> Tensor<T> Tensor<T>::slice(Shape s, Shape offset)
{
    size_t x = offset[0];
    size_t y = offset[1];
    size_t w = offset[2];
    size_t h = offset[3];
    return Tensor<T>(s, &data[x + y * dim_mul[0] + w * dim_mul[1] + h * dim_mul[2]]);
}

//
// copy part of another Tensor (src) to the this Tensor,
// with the specified shape and offsets.
//
template<typename T> void Tensor<T>::copyFrom(Tensor<T> &src, Shape rect, Shape src_offset, Shape dest_offset)
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
