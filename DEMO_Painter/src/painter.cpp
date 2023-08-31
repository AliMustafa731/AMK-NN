
#include "painter.h"
#include "utils/random.h"

template<typename T> T Max(T a, T b) { return a > b ? a : b; }
template<typename T> T Min(T a, T b) { return a < b ? a : b; }
template<typename T> T Abs(T a) { return a > 0 ? a : -a; }
template<typename T> void Swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

PainterNetwork::PainterNetwork(Shape map_shape) { map.init(map_shape.w, map_shape.h); }

void PainterNetwork::add(PaintElement *_element)
{
    elements.add(_element);
    parameters.add(&_element->parameters);
}

float* PainterNetwork::forward(float* input)
{
    for (auto n = elements.base ; n != NULL ; n = n->next)
    {
        PaintElement *e = n->value;
        e->forward(map);
    }

    return map.data;
}

void PainterNetwork::backward(Buffer<float> &d_map)
{
    for (auto n = elements.base; n != NULL; n = n->next)
    {
        PaintElement *e = n->value;
        e->backward(d_map);
    }
}


DrawLineElement::DrawLineElement()
{
    parameters = Parameter(4);

    for (int i = 0 ; i < parameters.size ; i++) { parameters.values[i] = random(0.0f, 100.0f); }
}

void DrawLineElement::forward(Buffer<float> &map)
{
    int x1 = (int)parameters.values[0];
    int y1 = (int)parameters.values[1];
    int x2 = (int)parameters.values[2];
    int y2 = (int)parameters.values[3];

    // clip the coordinates
    x1 = Max(0, Min(x1, map.w - 1));
    x2 = Max(0, Min(x2, map.w - 1));
    y1 = Max(0, Min(y1, map.h - 1));
    y2 = Max(0, Min(y2, map.h - 1));

    parameters.values[0] = (float)x1;
    parameters.values[1] = (float)y1;
    parameters.values[2] = (float)x2;
    parameters.values[3] = (float)y2;

    int dx = Abs(x1 - x2);
    int dy = Abs(y1 - y2);
    float dx_f = (float)dx;
    float dy_f = (float)dy;

    // make it left to right & up to down
    if (x1 > x2) { Swap(x1, x2); }
    if (y1 > y2) { Swap(y1, y2); }

    if (dx > dy)
    {
        for (int x = x1 ; x <= x2 ; x++)
        {
            float t = (float)(x - x1) / dx_f;
            int y = y1 + (int)(t * dy_f);

            map(x, y) = 1.0f;
        }
    }
    else
    {
        for (int y = y1; y <= y2; y++)
        {
            float t = (float)(y - y1) / dy_f;
            int x = x1 + (int)(t * dx_f);

            map(x, y) = 1.0f;
        }
    }
}

void DrawLineElement::backward(Buffer<float> &d_map)
{

}
