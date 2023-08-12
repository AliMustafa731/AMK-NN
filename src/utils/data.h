#pragma once

#include <cassert>
#include "common.h"
#include "utils/geometry.h"
#include "utils/array.h"


struct DataSet
{
	int sample_size, samples_num;
	Shape shape;
	Array<float> data;
	Array<float*> ptr;

	DataSet() {}
	~DataSet() { release(); }

	void init(Shape _shape, int _samples_num);
	void release();

	inline float* operator[](int i) const
	{
		AMK_ASSERT(i < ptr.size);
		return ptr[i];
	}
	inline float* &operator[](int i)
	{
		AMK_ASSERT(i < ptr.size);
		return ptr[i];
	}
};

struct Color
{
	unsigned char b, g, r, a;

	Color() {}
	Color(unsigned char _r, unsigned char _g, unsigned char _b)
	{
		r = _r;  g = _g;  b = _b;  a = 0;
	}
};

struct Colorf
{
	float r, g, b;

	Colorf() {}
	Colorf(float _r, float _g, float _b)
	{
		r = _r;  g = _g;  b = _b;
	}
};

void load_image(const char* filename, Color *dest, int *_w = NULL, int *_h = NULL);

void load_image(const char* filename, Array<Color> &dest, int *_w = NULL, int *_h = NULL);

void reverse_bytes(unsigned char* dest, unsigned char* src, int size);

void load_mnist_images(const char* filename, DataSet &dest, int samples = 0);

void load_mnist_labels(const char* filename, DataSet &dest, int num_class, int samples = 0);

__forceinline void embed_one_channel_to_color(Color *dest, unsigned char* src, int size)
{
	for (int i = 0; i < size; i++)
	{
		dest[i].r = src[i];
		dest[i].g = src[i];
		dest[i].b = src[i];
		dest[i].a = 0;
	}
}

__forceinline void rgb_to_float(Colorf *dest, Color *src, int size)
{
	for (int i = 0; i < size; i++)
	{
		dest[i].r = (float)src[i].r;
		dest[i].g = (float)src[i].g;
		dest[i].b = (float)src[i].b;
	}
}

__forceinline void float_to_rgb(Color *dest, Colorf *src, int size)
{
	for (int i = 0; i < size; i++)
	{
		src[i].r = (unsigned char)dest[i].r;
		src[i].g = (unsigned char)dest[i].g;
		src[i].b = (unsigned char)dest[i].b;
	}
}

__forceinline void rgb_to_float_one_channel(float *dest, unsigned char *src, int size)
{
	for (int i = 0; i < size; i++)
	{
		dest[i] = (float)src[i];
	}
}

__forceinline void float_to_rgb_one_channel(unsigned char *dest, float *src, int size)
{
	for (int i = 0; i < size; i++)
	{
		dest[i] = (unsigned char)src[i];
	}
}

__forceinline void float_one_channel_to_full_rgb(Color* dest, float* src, int size)
{
	Array<unsigned char> temp(size);

	float_to_rgb_one_channel(temp.data, src, size);
	embed_one_channel_to_color(dest, temp.data, size);

	temp.release();
}

template<typename T> void arrange_channels_to_blocks(Array<T> &dest, Array<T> &src, int channels)
{
	for (int j = 0; j < channels; j++)
	{
		for (int i = 0; i < dest.size / channels; i++)
		{
			dest[i * channels] = src[i + j];
		}
	}
}

template<typename T> void arrange_channels_to_color(Array<T> &dest, Array<T> &src, int channels)
{
	for (int j = 0; j < channels; j++)
	{
		for (int i = 0; i < dest.size / channels; i++)
		{
			dest[i + j] = src[i * channels];
		}
	}
}

template<typename T> __forceinline void make_ptr_list(Array<T*> &dest, Array<T> &src, int element_size)
{
	int elements_num = src.size / element_size;
	dest.init(elements_num);

	for (int i = 0; i < elements_num; i++)
	{
		dest[i] = &src[i * element_size];
	}
}

__forceinline float mean_value(float* data, int size)
{
	float mean = 0;

	for (int i = 0; i < size; i++)
	{
		mean += data[i];
	}

	mean = mean / (float)size;

	return mean;
}

__forceinline float standard_deviation(float* data, int size, float mean)
{
	float deviation = 0;

	for (int i = 0; i < size; i++)
	{
		float x = data[i] - mean;
		deviation += x * x;
	}

	deviation = deviation / (float)size;
	deviation = sqrt(deviation);

	return deviation;
}

__forceinline void standardize_data(float* data, int size)
{
	float mean = mean_value(data, size);
	float deviation = standard_deviation(data, size, mean);

	for (int i = 0; i < size; i++)
	{
		data[i] = (data[i] - mean) / (deviation + 1e-5f);
	}
}
