
#include "painter_network.h"
#include <iostream>

void PainterNetwork::draw()
{
	for (int i = 0; i < pens.size(); i++)
	{
		pens[i]->draw(Y);
	}

	for (int i = 0; i < Y.size; i++)
	{
		if (Y[i] > 255.0f)
		{
			Y[i] = 255.0f;
		}
	}
}

void PainterNetwork::fit(Buffer<float> &target)
{
	for (int i = 0; i < pens.size(); i++)
	{
		pens[i]->backward(dY);
	}
}

void PenLineDrawer::init(PainterNetwork *parent, int pos)
{
	X = parent->parameters.values.data + pos;
	dX = parent->parameters.gradients.data + pos;
}

void PenLineDrawer::draw(Buffer<float> &target)
{
	int x1 = (int)X[0];
	int y1 = (int)X[1];
	int x2 = (int)X[2];
	int y2 = (int)X[3];

	int dx = std::abs(x2 - x1);
	int dy = std::abs(y2 - y1);
	float dx_f = (float)dx;
	float dy_f = (float)dy;

	// make it left to right
	if (x1 > x2)
	{
		std::swap(x1, x2);
		std::swap(y1, y2);
	}

	if (dx > dy)
	{
		for (int x = x1 ; x < x2; x++)
		{
			float t = (float)(x - x1) / dx_f;

			int y = (1.0f - t)*y1 + t * y2;

			target(x, y) += 255.0f;
		}
	}
	else
	{
		for (int y = y1; y < y2; y++)
		{
			float t = (float)(y - y1) / dy_f;

			int x = (1.0f - t)*x1 + t * x2;

			target(x, y) += 255.0f;
		}
	}
}

void PenLineDrawer::backward(Buffer<float> &d_output)
{

}
