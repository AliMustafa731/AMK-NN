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
    Tensor<float> map, dX;

    PainterNetwork(){}
    PainterNetwork(Shape map_shape);

	void add(PaintElement *_element);
    Tensor<float>& forward(Tensor<float>& input);
    Tensor<float>& backward(Tensor<float>& d_map);
};
