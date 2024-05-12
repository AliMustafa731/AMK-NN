#pragma once

#include <Windows.h>
#include <CommCtrl.h>

#include <data/tensor.h>

struct Color
{
    uint8_t b, g, r, a;

    Color() {}
    Color(uint8_t _r, uint8_t _g, uint8_t _b) : r(_r), g(_g), b(_b), a(0) {}

    Color& operator+=(const Color& rhs) { return *this; }
    Color& operator-=(const Color& rhs) { return *this; }
    Color& operator*=(const Color& rhs) { return *this; }
};

struct Image
{
    BITMAPINFO info;
    Tensor<Color> img;

    Image();
    Image(int w, int h, Color* _data = NULL);

    void inti_from_float(Tensor<float>& src);

    void draw(HDC hdc, int x, int y, int w = 0, int h = 0);
};

