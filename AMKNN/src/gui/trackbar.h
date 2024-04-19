#pragma once

#include <Windows.h>
#include <CommCtrl.h>
#include <string>

struct TrackBar
{
    HWND win_handle, left_txt, right_txt, state_txt;
    float min_rng, max_rng;

    TrackBar();
    TrackBar(HWND parent, int x, int y, int w, int h, const char* l_txt,
        const char* r_txt, float _min_rng = 0.0f, float _max_rng = 1.0f);

    void update();
    void SetPos(float x);
    float GetPos();
};
