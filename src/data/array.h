#pragma once

#include <iostream>
#include <cassert>
#include "common.h"

//---------------------------
//   Static Array Structure
//---------------------------
template<typename T> struct Array
{
private:
    int _size, _length;

public:
	T* data;

	Array() { _size = 0;  _length = 0;  data = NULL; }
    Array(int __size) { init(__size); }
    Array(int __size, T* _data) { init(__size, _data); }

	inline int capacity() const { return _size; }
	inline int size() const { return _length; }

	inline T operator[](int i) const { AMK_ASSERT(i < _size);  return data[i]; }
	inline T &operator[](int i)      { AMK_ASSERT(i < _size);  return data[i]; }

    void init(int __size)
    {
		_size = __size;
		_length = _size;
		data = new T[_size];

		// initialize the memory to zero
        unsigned char* p = (unsigned char*)data;
		for (int i = 0; i < _size * sizeof(T); i++) { p[i] = 0; }
    }

    void init(int __size, T* _data)
    {
		_size = __size;
		_length = _size;
		data = _data;
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
		data[_length] = val;
		_length += 1;
	}
};

//----------------------------------------------------
//   Static Array, with 2D meta-info & accesors
//----------------------------------------------------
template<typename T> struct Buffer
{
    T *data;
    int w, h, size;

	Buffer() { w = 0;  h = 0;  size = 0;  data = NULL; }
    Buffer(int _w, int _h) { init(_w, _h); }
    Buffer(int _w, int _h, T* _data) { init(_w, _h, _data); }

	inline T operator()(int x, int y) const { AMK_ASSERT(x < w && y < h);  return data[x + y * w]; }
	inline T &operator()(int x, int y)      { AMK_ASSERT(x < w && y < h);  return data[x + y * w]; }
	inline T operator[](int i) const        { AMK_ASSERT(i < size);  return data[i]; }
	inline T &operator[](int i)             { AMK_ASSERT(i < size);  return data[i]; }

    void init(int _w, int _h)
    {
        w = _w;
        h = _h;
        size = w * h;
        data = new T[w * h];

		// initialize the memory to zero
		unsigned char* p = (unsigned char*)data;
		for (int i = 0; i < size * sizeof(T); i++) { p[i] = 0; }
    }

    void init(int _w, int _h, T* _data)
    {
        w = _w;
        h = _h;
        size = w * h;
        data = _data;
    }

    void release()
    {
        if (data != NULL)
        {
            delete[] data;
            data = NULL;
			size = 0;
			w = 0;
			h = 0;
        }
    }
};

//----------------------------------
//   Double Linked-List
//----------------------------------
template<typename T> struct List
{
	struct Node
	{
		T value;
		Node *next, *prev;

		Node()
		{
			next = NULL;
			prev = NULL;
		}
		Node(T val)
		{
			value = val;
			next = NULL;
			prev = NULL;
		}
	};

	Node *base, *head;
	int length;

	List() { length = 0; }

	void add(T val)  // add node at the end of the list
	{
		if (length < 1)  // the first addition
		{
			base = new Node(val);
			head = base;
		}
		else
		{
			head->next = new Node(val);
			head->next->prev = head;
			head = head->next;
		}

		length += 1;
	}

	void remove(Node *node)
	{
		bool external_node = true;

		for (Node *n = base ; n->next != NULL ; n = n->next)
		{
			// check if the node is within our linked list
			if (n == node)
			{
				external_node = false;
				break;
			}
		}
		if (external_node || node == NULL) { return; }


		if (node->prev == NULL)  // the first node
		{
			Node *temp = base;

			if (base->next != NULL)
			{
				base->next->prev = NULL;
				base = base->next;
			}
			else
			{
				base = NULL;
			}

			delete temp;
		}
		else if (node->next == NULL)  // the last node
		{
			Node *temp = head;

			head->prev->next = NULL;
			head = head->prev;

			delete temp;
		}
		else  // intermediate node
		{
			node->next->prev = node->prev;
			node->prev->next = node->next;
			delete node;
		}

		length -= 1;
	}

	void release()
	{
		Node *n = base;

		while (n != NULL)
		{
			Node *temp = n;
			n = n->next;
			delete temp;
		}

		length = 0;
	}
};
