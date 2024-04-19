
#include "trackbar.h"

TrackBar::TrackBar() {}

TrackBar::TrackBar(HWND parent, int x, int y, int w, int h, const char* l_txt,
                   const char* r_txt, float _min_rng, float _max_rng)
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

void TrackBar::update()
{
    SetWindowText(state_txt, std::to_string(GetPos()).c_str());
}

float TrackBar::GetPos()
{
    return min_rng + (max_rng - min_rng) * ((float)SendMessage(win_handle, TBM_GETPOS, 0, 0) / 10000.0f);
}

void TrackBar::SetPos(float x)
{
    SendMessage(win_handle, TBM_SETPOS, TRUE, (LPARAM) 10000.0f * (x - min_rng) / (max_rng - min_rng));
    update();
}
