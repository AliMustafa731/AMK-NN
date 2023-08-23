#pragma once

#include <Windows.h>
#include <CommCtrl.h>
#include <string>
#include "data/array.h"
#include "data/buffer.h"
#include "data/dataset.h"
#include "utils/graphics.h"
#include "common.h"

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

struct Program
{
    int width, height;
    HWND win_handle;

    Program(){}
    Program(const char* name, int _w, int _h)
    {
        init(name, _w, _h);
    }

    void init(const char* name, int _w, int _h);
};

struct Image
{
    BITMAPINFO info;
    Buffer<Color> img;

    Image() {}
    Image(int w, int h, Color* _data = NULL)
    {
        info.bmiHeader.biSize = sizeof(info.bmiHeader);
        info.bmiHeader.biWidth = w;
        info.bmiHeader.biHeight = h;
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB;

        if (_data == NULL)
        {
            img.init(w, h);
        }
        else
        {
            img.init(w, h, _data);
        }
    }

    void draw(HDC hdc, int x, int y, int w = 0, int h = 0)
    {
        if (w == 0) { w = info.bmiHeader.biWidth; }
        if (h == 0) { h = info.bmiHeader.biHeight; }

        StretchDIBits
        (
            hdc, x, y, w, h, 0, 0,
            img.w, img.h, (void*)img.data,
            &info, DIB_RGB_COLORS, SRCCOPY
        );
    }
};

struct TrackBar
{
    HWND win_handle, left_txt, right_txt, state_txt;
    float min_rng, max_rng;

    TrackBar(){}
    TrackBar(HWND parent, int x, int y, int w, int h, const char* l_txt, const char* r_txt, float _min_rng = 0.0f, float _max_rng = 1.0f)
    {
        min_rng = _min_rng;
        max_rng = _max_rng;

        win_handle = CreateWindow
        (
            TRACKBAR_CLASS, "", WS_VISIBLE | WS_CHILD | TBS_ENABLESELRANGE | TBS_AUTOTICKS,
            x, y, w, h, parent, NULL, GetModuleHandle(NULL), NULL
        );

        right_txt = CreateWindow
        (
            "STATIC", r_txt, WS_VISIBLE | WS_CHILD | SS_LEFT,
            0, 0, 150, 30, parent, NULL, GetModuleHandle(NULL), NULL
        );
        left_txt = CreateWindow
        (
            "STATIC", l_txt, WS_VISIBLE | WS_CHILD | SS_LEFT,
            0, 0, 50, 30, parent, NULL, GetModuleHandle(NULL), NULL
        );
        state_txt = CreateWindow
        (
            "STATIC", l_txt, WS_VISIBLE | WS_CHILD | SS_LEFT,
            x + w + 130, y - 5, 65, 30, parent, NULL, GetModuleHandle(NULL), NULL
        );

        SendMessage(win_handle, TBM_SETRANGE, TRUE, MAKELONG(0, 10000));
        SendMessage(win_handle, TBM_SETPOS, 0, 0);
        SendMessage(win_handle, TBM_SETTICFREQ, 10, 0);
        SendMessage(win_handle, TBM_SETPAGESIZE, 0, (LPARAM)10);
        SendMessage(win_handle, TBM_SETBUDDY, FALSE, (LPARAM)right_txt);
        SendMessage(win_handle, TBM_SETBUDDY, TRUE, (LPARAM)left_txt);
    }

    void update()
    {
        SetWindowText(state_txt, std::to_string(get_pos()).c_str());
    }

    float get_pos()
    {
        return min_rng + (max_rng - min_rng) * ((float)SendMessage(win_handle, TBM_GETPOS, 0, 0) / 10000.0f);
    }

    void set_pos(float x)
    {
        SendMessage(win_handle, TBM_SETPOS, TRUE, (LPARAM) 10000.0f * (x - min_rng) / (max_rng - min_rng));
        update();
    }
};
