
#include <Windows.h>
#include "program.h"


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, int nCmdShow)
{
    Program program;

    program.init("AMK Painter", 960, 640);

    return 0;
}
