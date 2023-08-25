
#include <Windows.h>
#include <CommCtrl.h>
#include <fstream>
#include <string>
#include <ctime>

#include <process.h>
#include "program.h"
#include "loaders.h"

#include "data/dataset.h"
#include "utils/graphics.h"
#include "utils/utils.h"
#include "neural_network.h"

// enable windows visual theme style
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

//-----------------------------------------------------
//    All code that controls and manage the window
//-----------------------------------------------------

void Program::init(const char* name, int _w, int _h)
{
    srand(time(0)); // seed the random generator

    // create and register the window
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "AMKNN";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClass(&wc);

    win_handle = CreateWindowEx
    (
        0, "AMKNN", name,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT,
        _w, _h, NULL, NULL, GetModuleHandle(NULL), NULL
    );

    if (win_handle == NULL)
    {
        MessageBox(NULL, "Error : can't initialize the program : \"win_handle\"", "Opss!", MB_OK);
        exit(0);
    }

    ShowWindow(win_handle, SW_SHOW);
    UpdateWindow(win_handle);

    INITCOMMONCONTROLSEX icex;
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_LISTVIEW_CLASSES;
    InitCommonControlsEx(&icex);

    // run messages loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

//-----------------------------------------------------
//    Main Window Callback
//-----------------------------------------------------

void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void onCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CREATE:
    {
        onCreate(hwnd, uMsg, wParam, lParam);

    } break;

    case WM_COMMAND:
    {
        onCommand(hwnd, uMsg, wParam, lParam);

    } break;

    case WM_PAINT:
    {
        onDraw(hwnd, uMsg, wParam, lParam);

    } break;

    case WM_DESTROY:
    {
        PostQuitMessage(0);

    } break;

    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//-----------------------------------------------------
//    Training & Logic of the program
//-----------------------------------------------------


void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{

}

void onCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{

}

void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);

    FillRect(hdc, &ps.rcPaint, (HBRUSH)COLOR_WINDOW);

    EndPaint(hwnd, &ps);
    ReleaseDC(hwnd, hdc);
}

