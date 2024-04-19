#pragma once

#include <Windows.h>
#include <CommCtrl.h>

#include "image.h"
#include "trackbar.h"

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

struct Program
{
    HWND win_handle;

    Program();
    Program(const char* name, int _w, int _h);

    void start(const char* name, int _w, int _h);
    virtual void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {};
    virtual void onCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {};
    virtual void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {};
    virtual void onHScroll(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {};
};

void CreateOPENFILENAME(OPENFILENAME *ofn, HWND parent, char* buffer, int buffer_size,
                        const char* filters, const char* default_ext, DWORD flags);
