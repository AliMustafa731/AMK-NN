#pragma once

#include "layers/base_layer.h"

struct SigmoidLayer : NeuralLayer
{
	SigmoidLayer();

	void init(Shape _in_shape);
	void release() {}
	float* forward(float* input);
	float* backward(float* d_output);
	void save(std::ofstream& file) {}
	void load(std::ifstream& file) {}
};
