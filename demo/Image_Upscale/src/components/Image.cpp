
#include "Image.h"

Image::Image(wxWindow* parent, int width, int height, int view_width, int view_height, bool clear)
{
    m_image = wxImage(width, height, clear);
    m_image = m_image.Scale(view_width, view_height, wxIMAGE_QUALITY_NEAREST);
    m_bitmap = wxBitmap(m_image);
    m_static_bitmap = new wxStaticBitmap(parent, wxID_ANY, m_bitmap);
}

wxStaticBitmap* Image::getBitmapHandle()
{
    return m_static_bitmap;
}
