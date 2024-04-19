#pragma once

#include <Windows.h>
#include <CommCtrl.h>

#include <data/tensor.h>
#include <utils/graphics.h>

struct Image
{
    BITMAPINFO info;
    Tensor<Color> img;

    Image();
    Image(int w, int h, Color* _data = NULL);

    void draw(HDC hdc, int x, int y, int w = 0, int h = 0);
};

