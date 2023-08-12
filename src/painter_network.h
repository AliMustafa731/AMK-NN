#pragma once

#include <vector>
#include "utils/array.h"
#include "optimizers.h"
#include "utils/data.h"

struct PainterNetwork;

struct Penciel
{
	int param_size;

	Penciel(){}

	virtual void init(PainterNetwork *parent, int pos) = 0;
	virtual void draw(Buffer<float> &target) = 0;
	virtual void backward(Buffer<float> &d_output) = 0;
};

struct PenLineDrawer : Penciel
{
	float *X, *dX;

	PenLineDrawer()
	{
		param_size = 4;
	}

	void init(PainterNetwork *parent, int pos);
	void draw(Buffer<float> &target);
	void backward(Buffer<float> &d_output);
};

struct PainterNetwork
{
	Buffer<float> Y, dY;
	std::vector<Penciel*> pens;
	Parameter parameters;

	PainterNetwork(){}
	PainterNetwork(int w, int h, std::vector<Penciel*> _pens)
	{
		Y.init(w, h);
		dY.init(w, h);
		pens = _pens;

		int _size = 0;

		for (int i = 0; i < pens.size(); i++)
		{
			_size += pens[i]->param_size;
		}
		parameters = Parameter(_size);

		int pos = 0;

		for (int i = 0; i < pens.size(); i++)
		{
			pens[i]->init(this, pos);

			pos += pens[i]->param_size;
		}
	}

	void draw();
	void fit(Buffer<float> &target);
};
