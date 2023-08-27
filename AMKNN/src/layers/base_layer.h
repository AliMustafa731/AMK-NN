#pragma once

#include "data/array.h"
#include "optimizers/parameter.h"
#include "utils/geometry.h"
#include <fstream>

struct NeuralLayer
{
	Array<float> X, dX, Y, dY;
	Array<Parameter> parameters;
	int in_size, out_size, type;
	Shape in_shape, out_shape;
	bool trainable;

	NeuralLayer() {}

	void allocate(int _in_size, int _out_size);
	void deallocate();
	void save_parameters(std::ofstream& file);
	void load_parameters(std::ifstream& file);
	void setTrainable(bool state);

	virtual void init(Shape _in_shape) = 0;
	virtual void release() = 0;
	virtual float* forward(float* input) = 0;
	virtual float* backward(float* d_output) = 0;
	virtual void save(std::ofstream& file) = 0;
	virtual void load(std::ifstream& file) = 0;
};
