#pragma once

#include "optimizers/parameter.h"
#include "data/array.h"
#include "data/list.h"
#include "data/buffer.h"
#include "utils/geometry.h"


struct PaintElement
{
    Parameter parameters;

	PaintElement(){}
	virtual ~PaintElement(){}

	virtual void forward(Buffer<float> &map) = 0;
	virtual void backward(Buffer<float> &d_map) = 0;
};

struct DrawLineElement : PaintElement
{
    DrawLineElement();

    void forward(Buffer<float> &map);
    void backward(Buffer<float> &d_map);
};

struct PainterNetwork
{
    List<PaintElement*> elements;
    List<Parameter*> parameters;
    Buffer<float> map;

    PainterNetwork(){}
    PainterNetwork(Shape map_shape);

	void add(PaintElement *_element);
	float* forward(float* input);
    void   backward(Buffer<float> &d_map);
};
