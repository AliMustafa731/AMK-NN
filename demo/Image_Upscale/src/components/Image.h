#pragma once

#include <wx/wx.h>

class Image
{
public:
    Image(wxWindow* parent, int width, int height, int view_width, int view_height, bool clear=true);
    wxStaticBitmap* getBitmapHandle();

private:
    wxImage m_image;
    wxBitmap m_bitmap;
    wxStaticBitmap* m_static_bitmap;
};
