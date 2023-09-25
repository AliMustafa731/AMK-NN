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
    int _size, _length;

public:
    T* data;

    Array() { _size = 0;  _length = 0;  data = NULL; }
    Array(int __size, T* _data = NULL) { init(__size, _data); }

    inline int capacity() const { return _size; }
    inline int size() const { return _length; }

    inline T operator[](int i) const { AMK_ASSERT(i < _size);  return data[i]; }
    inline T &operator[](int i)      { AMK_ASSERT(i < _size);  return data[i]; }

    void init(int __size, T* _data = NULL)
    {
        _size = __size;
        _length = _size;

        if (_data != NULL)
        {
            data = _data;
        }
        else
        {
            data = new T[_size];
            std::memset(data, 0, _size * sizeof(T));
        }
    }

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
            _size = 0;
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
        if (data == NULL) { return; }

        // if length eceeds the capacity
        // realloctate double the amount of memory
        if (_length >= _size)
        {
            _size *= 2;

            // initialize & copy new buffer
            T* tmp = new T[_size];
            for (int i = 0; i < _length; i++) { tmp[i] = data[i]; }

            delete[] data;
            data = tmp;
        }

        data[_length] = val;
        _length += 1;
    }
};
