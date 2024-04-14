#pragma once

#include <optimizers/parameter.h>
#include <data/array.h>
#include <data/list.h>
#include <data/tensor.h>
#include <utils/geometry.h>


struct PaintElement
{
    Parameter parameters;

    PaintElement(){}
    virtual ~PaintElement(){}

    virtual void forward(Tensor<float> &map) = 0;
    virtual void backward(Tensor<float> &d_map) = 0;
};

struct DrawLineElement : PaintElement
{
    DrawLineElement();

    void forward(Tensor<float> &map);
    void backward(Tensor<float> &d_map);
};

struct PainterNetwork
{
    List<PaintElement*> elements;
    List<Parameter*> parameters;
    Tensor<float> map, d_map;

    PainterNetwork(){}
    PainterNetwork(Shape map_shape);

    void add(PaintElement *_element);
    void forward();
    void backward();
};

// utilites
template<typename T> T Max(T a, T b) { return a > b ? a : b; }
template<typename T> T Min(T a, T b) { return a < b ? a : b; }
template<typename T> T Abs(T a) { return a > 0 ? a : -a; }
template<typename T> void Swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}
