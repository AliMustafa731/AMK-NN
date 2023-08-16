
#include "utils/data.h"
#include "utils/stb_image.h"
#include <Windows.h>
#include <fstream>

void DataSet::init(Shape _shape, int _samples_num)
{
    shape = _shape;
    samples_num = _samples_num;
    sample_size = shape.size();

    data.init(samples_num * sample_size);
    make_ptr_list<float>(ptr, data, sample_size);
}

void DataSet::release()
{
    sample_size = 0;
    samples_num = 0;
    data.release();
    ptr.release();
}

bool load_image(const char* filename, Color *dest, int *_w, int *_h)
{
    int w, h;
    unsigned char* _data = stbi_load(filename, &w, &h, NULL, 3);

    if (_data == NULL)
    {
        MessageBox(NULL, "Error : can't load the image", "Opss!", MB_OK);
        return false;
    }

    if (_w != NULL) { *_w = w; }
    if (_h != NULL) { *_h = h; }

    dest = new Color[w * h];

    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            int _idx = x + y * w;
            int _idx_flip = x + (h - 1 - y)*w;
            dest[_idx].r = _data[((_idx_flip) * 3)];
            dest[_idx].g = _data[((_idx_flip) * 3) + 1];
            dest[_idx].b = _data[((_idx_flip) * 3) + 2];
        }
    }

    delete[] _data;

    return true;
}

bool load_image(const char* filename, Array<Color> &dest, int *_w, int *_h)
{
    int w, h;
    unsigned char* _data = stbi_load(filename, &w, &h, NULL, 3);

    if (_data == NULL)
    {
        MessageBox(NULL, "Error : can't load the image", "Opss!", MB_OK);
        return false;
    }

    if (_w != NULL) { *_w = w; }
    if (_h != NULL) { *_h = h; }

    dest.init(w * h);

    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            int _idx = x + y * w;
            int _idx_flip = x + (h - 1 - y)*w;
            dest[_idx].r = _data[((_idx_flip) * 3)];
            dest[_idx].g = _data[((_idx_flip) * 3) + 1];
            dest[_idx].b = _data[((_idx_flip) * 3) + 2];
        }
    }

    delete[] _data;

    return true;
}

void reverse_bytes(unsigned char* dest, unsigned char* src, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = src[size - i - 1];
    }
}

void encode_one_hot(Array<float> &dest, Array<int> &src, int num_class)
{
    for (int i = 0; i < src.size; i++)
    {
        dest[i * num_class + src[i]] = 1.0f;
    }
}

void encode_one_hot(Array<float> &dest, Array<unsigned char> &src, int num_class)
{
    for (int i = 0; i < src.size; i++)
    {
        dest[i * num_class + src[i]] = 1.0f;
    }
}

bool load_mnist_images(const char* filename, DataSet &dest, int samples)
{
    int samples_count, _samples_count, width, _width, height, _height;

    std::ifstream file;
    file.open(filename, std::ios::out | std::ios::binary);

    if (file.fail()) return false;

    file.seekg(4);
    file.read((char*)&_samples_count, sizeof(int));
    file.read((char*)&_width, sizeof(int));
    file.read((char*)&_height, sizeof(int));

    reverse_bytes((unsigned char*)&width, (unsigned char*)&_width, sizeof(int));
    reverse_bytes((unsigned char*)&height, (unsigned char*)&_height, sizeof(int));

    if (samples == 0)
    {
        reverse_bytes((unsigned char*)&samples_count, (unsigned char*)&_samples_count, sizeof(int));
    }
    else
    {
        samples_count = samples;
    }

    int size = samples_count * width * height;

    Array<unsigned char> temp(size);

    file.read((char*)temp.data, size * sizeof(char));

    dest.init(Shape(width, height, 1), samples_count);

    // copy and flipp vertically
    for (int i = 0; i < dest.samples_num; i++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                dest.data[x + y * width + i * width * height] = (float)temp[x + (height - 1 - y) * width + i * width * height];
            }
        }
    }

    temp.release();

    file.close();

    return true;
}

bool load_mnist_labels(const char* filename, DataSet &dest, int num_class, int samples)
{
    int samples_count, _samples_count;

    std::ifstream file;
    file.open(filename, std::ios::out | std::ios::binary);

    if (file.fail()) return false;

    file.seekg(4);
    file.read((char*)&_samples_count, sizeof(int));

    if (samples == 0)
    {
        reverse_bytes((unsigned char*)&samples_count, (unsigned char*)&_samples_count, sizeof(int));
    }
    else
    {
        samples_count = samples;
    }

    Array<unsigned char> temp(samples_count);
    file.read((char*)temp.data, samples_count * sizeof(char));

    dest.init(Shape(num_class, 1, 1), samples_count);

    encode_one_hot(dest.data, temp, num_class);

    temp.release();

    file.close();

    return true;
}
