
#include <painter.h>
#include <utils/random.h>


PainterNetwork::PainterNetwork(Shape map_shape)
{
    map.init(map_shape.w, map_shape.h, map_shape.d);
}

void PainterNetwork::add(PaintElement *_element)
{
    elements.add(_element);
    parameters.add(&_element->parameters);
}

void PainterNetwork::forward()
{
    for (auto n = elements.base ; n != NULL ; n = n->next)
    {
        PaintElement *e = n->value;
        e->forward(map);
    }
}

void PainterNetwork::backward()
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

void DrawLineElement::forward(Tensor<float>& map)
{
    int x1 = (int)parameters.values[0];
    int y1 = (int)parameters.values[1];
    int x2 = (int)parameters.values[2];
    int y2 = (int)parameters.values[3];

    // clip the coordinates
    x1 = Max(0, Min(x1, map.s.w - 1));
    x2 = Max(0, Min(x2, map.s.w - 1));
    y1 = Max(0, Min(y1, map.s.h - 1));
    y2 = Max(0, Min(y2, map.s.h - 1));

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

void DrawLineElement::backward(Tensor<float>& d_map)
{

}
