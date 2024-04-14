
#include <loaders.h>
#include <stb_image.h>
#include <fstream>

int readImage(const char* filename, Tensor<Color> &target)
{
    int _w, _h;
    unsigned char* _data = stbi_load(filename, &_w, &_h, NULL, 3);
    if (_data == NULL) return 0;

    target.release();  // make sure it's empty
    target.init(_w, _h);

    // copy and flip vertically
    for (int x = 0; x < target.s.w; x++)
    {
        for (int y = 0; y < target.s.h; y++)
        {
            int _idx = x + y * target.s.w;
            int _idx_flip = x + ((target.s.h - 1) - y)*target.s.w;
            target[_idx].r = _data[((_idx_flip) * 3)];
            target[_idx].g = _data[((_idx_flip) * 3) + 1];
            target[_idx].b = _data[((_idx_flip) * 3) + 2];
        }
    }
    delete[] _data;

    return 1;
}

int readImage(const char* filename, Tensor<Colorf> &target)
{
    int _w, _h;
    unsigned char* _data = stbi_load(filename, &_w, &_h, NULL, 3);
    if (_data == NULL) return 0;

    target.release();  // make sure it's empty
    target.init(_w, _h);

    // copy and flip vertically
    for (int x = 0; x < target.s.w; x++)
    {
        for (int y = 0; y < target.s.h; y++)
        {
            int _idx = x + y * target.s.w;
            int _idx_flip = x + ((target.s.h - 1) - y)*target.s.w;
            target[_idx].r = (float)_data[((_idx_flip) * 3)];
            target[_idx].g = (float)_data[((_idx_flip) * 3) + 1];
            target[_idx].b = (float)_data[((_idx_flip) * 3) + 2];
        }
    }
    delete[] _data;

    return 1;
}

void normalize(Tensor<Colorf> &target)
{
    for (int i = 0; i < target.size(); i++)
    {
        target[i].r /= 255.0f;
        target[i].g /= 255.0f;
        target[i].b /= 255.0f;
    }
}

void normalize(Tensor<float> &target)
{
    for (int i = 0; i < target.size(); i++)
    {
        target[i] /= 255.0f;
    }
}

void denormalize(Tensor<Colorf> &target)
{
    for (int i = 0; i < target.size(); i++)
    {
        target[i].r *= 255.0f;
        target[i].g *= 255.0f;
        target[i].b *= 255.0f;
    }
}

void denormalize(Tensor<float> &target)
{
    for (int i = 0; i < target.size(); i++)
    {
        target[i] *= 255.0f;
    }
}
