#pragma once

#include <cstring>
#include <cassert>
#include <common.h>

//---------------------------------
//   Dynamic Array Structure
//   you must use "reserve()"
//   if you want to add elements
//---------------------------------
template<typename T> struct Array
{
private:
    int _capacity, _length;

public:
    T* data;

    Array() : _capacity(0), _length(0), data(NULL) {}
    Array(int __capacity, T* _data = NULL) { init(__capacity, _data); }

    inline int capacity() const { return _capacity; }
    inline int size() const { return _length; }

    inline T operator[](int i) const { AMK_ASSERT(i < _capacity);  return data[i]; }
    inline T &operator[](int i)      { AMK_ASSERT(i < _capacity);  return data[i]; }

    void init(int __capacity, T* _data = NULL)
    {
        _capacity = __capacity;
        _length = _capacity;

        if (_data != NULL)
        {
            data = _data;
        }
        else
        {
            data = new T[_capacity];
            std::memset(data, 0, _capacity * sizeof(T));
        }
    }

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
            _capacity = 0;
            _length = 0;
        }
    }

    void resize(int __size)
    {
        release();
        init(__size);
    }

    void reserve(int __size)
    {
        release();
        init(__size);
        _length = 0;
    }

    void add(T val)
    {
        if (data == NULL)
        {
            reserve(5);
        }

        // if length eceeds the capacity
            // realloctate double the amount of memory
        if (_length >= _capacity)
        {
            // initialize & copy new buffer
            _capacity *= 2;
            T* tmp = new T[_capacity];

            for (int i = 0; i < _length; i++) { tmp[i] = data[i]; }

            delete[] data;
            data = tmp;
        }

        data[_length] = val;
        _length += 1;
    }
};
