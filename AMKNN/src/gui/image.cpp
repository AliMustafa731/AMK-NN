
#include "image.h"
#include <common.h>

Image::Image() {}

Image::Image(int w, int h, Color* _data)
{
    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biWidth = w;
    info.bmiHeader.biHeight = h;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;

    img.init(Shape(w, h), _data);
}

void Image::draw(HDC hdc, int x, int y, int w, int h)
{
    if (w == 0) { w = info.bmiHeader.biWidth; }
    if (h == 0) { h = info.bmiHeader.biHeight; }

    StretchDIBits
    (
        hdc, x, y, w, h, 0, 0,
        img.shape[0], img.shape[1], (void*)img.data,
        &info, DIB_RGB_COLORS, SRCCOPY
    );
}

void Image::inti_from_float(Tensor<float>& src)
{
    //***************************************************************
    // (width) and (height) must be equal,
    // (src) is either one-channel (gray) or three-channels (rgb)
    //***************************************************************

    AMK_ASSERT(img.shape[0] == src.shape[0] && img.shape[1] == src.shape[1] &&
               (src.shape[2] == 1 || src.shape[2] == 3) && img.shape[2] == 1);

    if (src.shape[2] == 1)
    {
        for (int i = 0; i < src.size(); i++)
        {
            img[i].r = (uint8_t)src[i];
            img[i].g = (uint8_t)src[i];
            img[i].b = (uint8_t)src[i];
            img[i].a = 0;
        }
    }
    else if (src.shape[2] == 3)
    {
        for (int x = 0; x < src.shape[0]; x++)
        {
            for (int y = 0; y < src.shape[1]; y++)
            {
                img(x, y).r = (uint8_t)src(x, y, 0);
                img(x, y).g = (uint8_t)src(x, y, 1);
                img(x, y).b = (uint8_t)src(x, y, 2);
                img(x, y).a = 0;
            }
        }
    }
}
