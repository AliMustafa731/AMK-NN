
#include "program.h"

#include <string>
#include <ctime>

//-----------------------------------------------------
//    Initilize The Program
//-----------------------------------------------------

static Program* mainProgram;

Program::Program(){}

Program::Program(const char* name, int _w, int _h)
{
    start(name, _w, _h);
}

void Program::start(const char* name, int _w, int _h)
{
    mainProgram = this;

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

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CREATE:
    {
        mainProgram->onCreate(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_COMMAND:
    {
        mainProgram->onCommand(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_HSCROLL:
    {
        mainProgram->onHScroll(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_PAINT:
    {
        mainProgram->onDraw(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_DESTROY:
    {
        PostQuitMessage(0);

    } return 0;

    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//-----------------------------------------------------
//    Others
//-----------------------------------------------------

void CreateOPENFILENAME(OPENFILENAME *ofn, HWND parent, char* buffer, int buffer_size,
                        const char* filters, const char* default_ext, DWORD flags)
{
    ZeroMemory(ofn, sizeof(OPENFILENAME));

    ofn->lStructSize = sizeof(OPENFILENAME);
    ofn->hwndOwner = parent;
    ofn->lpstrFile = (LPSTR)buffer;
    ofn->lpstrFile[0] = '\0';
    ofn->nMaxFile = buffer_size;
    ofn->lpstrFilter = (LPSTR)filters;
    ofn->nFilterIndex = 1;
    ofn->lpstrFileTitle = NULL;
    ofn->nMaxFileTitle = 0;
    ofn->lpstrInitialDir = NULL;
    ofn->lpstrDefExt = (LPSTR)default_ext;
    ofn->Flags = flags;
}
