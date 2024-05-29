#pragma once

#include <cstring>
#include <cassert>
#include <common.h>

/*
 * Dynamic Array Structure
 */
template<typename T> class Array
{
public:

    Array() : _capacity(0), _size(0), _data(NULL) {}

    Array(size_t __capacity, T* __data = NULL) { init(__capacity, __data); }

    /// accessors

    /// @return capacity of the array
    inline size_t capacity() const { return _capacity; }

    /// @return number of elements
    inline size_t size() const { return _size; }

    /// @return pointer to the memory of the array
    inline T* data() const { return _data; }

    /// setters and getters for array access (read/write)
    inline T operator[](size_t i) const { AMK_ASSERT(i < _capacity);  return _data[i]; }
    inline T &operator[](size_t i) { AMK_ASSERT(i < _capacity);  return _data[i]; }

    /*
     * initialize the array with the specified size and data pointer 
     * 
     * @param __size : size of the array
     * 
     * @param __data :
     * pointer to the data, if "NULL" (not passed),
     * new memory is allocated
     */
    void init(size_t __size, T* __data = NULL)
    {
        this->_capacity = __size;
        this->_size = __size;

        if (__data != NULL)
        {
            this->_data = __data;
        }
        else
        {
            this->_data = new T[this->_capacity];
            std::memset(this->_data, 0, this->_capacity * sizeof(T));
        }
    }

    /// release the memory of the array
    void release()
    {
        if (_data != NULL)
        {
            delete[] _data;
            _data = NULL;
            _capacity = 0;
            _size = 0;
        }
    }

    /*
     * re-allocate a memory with new size and initialize it with empty elemnts.
     *
     * @param __size : new size of the array
     */
    void resize(size_t __size)
    {
        release();
        init(__size);
    }

    /*
     * only re-allocate a memory with new size,
     * not filled with empty elemnts. (you can add elemnts to the array)
     *
     * @param __size : new size of the array
     */
    void reserve(size_t __size)
    {
        release();
        init(__size);
        _size = 0;
    }

    /// add an element to the end of the array
    /// @param val : element to be added
    void add(T val)
    {
        if (_data == NULL)
        {
            reserve(5);
        }

        // if length eceeds the capacity
            // realloctate double the amount of memory
        if (_size >= _capacity)
        {
            // initialize & copy new buffer
            _capacity *= 2;
            T* tmp = new T[_capacity];

            for (int i = 0; i < _size; i++) { tmp[i] = _data[i]; }

            delete[] _data;
            _data = tmp;
        }

        _data[_size] = val;
        _size++;
    }

protected:
    /// pointer to the memory where elements is to be stored
    T* _data;

    /// amount of memory allocated
    size_t _capacity;

    /// number of elements in the array
    size_t _size;
};
