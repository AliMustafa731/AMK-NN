
#include "image.h"

Image::Image() {}

Image::Image(int w, int h, Color* _data)
{
    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biWidth = w;
    info.bmiHeader.biHeight = h;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;

    img.init(w, h, 1, _data);
}

void Image::draw(HDC hdc, int x, int y, int w, int h)
{
    if (w == 0) { w = info.bmiHeader.biWidth; }
    if (h == 0) { h = info.bmiHeader.biHeight; }

    StretchDIBits
    (
        hdc, x, y, w, h, 0, 0,
        img.s.w, img.s.h, (void*)img.data,
        &info, DIB_RGB_COLORS, SRCCOPY
    );
}
