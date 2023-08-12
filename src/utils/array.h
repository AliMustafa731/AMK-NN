#pragma once

#include <iostream>
#include <cassert>
#include "common.h"

template<typename T> struct Array
{
	T* data;
	uint32_t size;

	Array() {}
	Array(uint32_t _size) { init(_size); }
	Array(uint32_t _size, T* _data) { init(_size, _data); }

	void init(uint32_t _size)
	{
		size = _size;
		data = new T[size];

		unsigned char* p = (unsigned char*)data;

		for (int i = 0; i < size * sizeof(T); i++)
		{
			p[i] = 0;
		}
	}

	void init(uint32_t _size, T* _data)
	{
		size = _size;
		data = _data;
	}

	void release()
	{
		if (data != NULL)
		{
			delete[] data;
			data = NULL;
		}
	}

	void resize(uint32_t _size)
	{
		release();
		init(_size);
	}

	inline T operator[](int i) const
	{
		AMK_ASSERT(i < size);
		return data[i];
	}
	inline T &operator[](int i)
	{
		AMK_ASSERT(i < size);
		return data[i];
	}
};

template<typename T> struct Buffer
{
	T *data;
	int w, h, size;

	Buffer() {}
	Buffer(int _w, int _h, T* _data)
	{
		w = _w;
		h = _h;
		size = w * h;
		data = _data;
	}
	Buffer(int _w, int _h) { init(_w, _h); }

	inline T operator()(int x, int y) const
	{
		AMK_ASSERT(x < w && y < h);
		return data[x + y * w];
	}
	inline T &operator()(int x, int y)
	{
		AMK_ASSERT(x < w && y < h);
		return data[x + y * w];
	}
	inline T operator[](int i) const
	{
		AMK_ASSERT(i < size);
		return data[i];
	}
	inline T &operator[](int i)
	{
		AMK_ASSERT(i < size);
		return data[i];
	}

	void init(int _w, int _h)
	{
		w = _w;
		h = _h;
		size = w * h;
		data = new T[w * h];

		unsigned char* p = (unsigned char*)data;

		for (int i = 0; i < size * sizeof(T); i++)
		{
			p[i] = 0;
		}
	}

	void release()
	{
		if (data != NULL)
		{
			delete[] data;
			data = NULL;
		}
	}
};
