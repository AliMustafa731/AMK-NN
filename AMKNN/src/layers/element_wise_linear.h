#pragma once

#include "layers/base_layer.h"

//----------------------------------------------
//  Element-wise linear sacle & offset
//----------------------------------------------
struct EltwiseLinear : NeuralLayer
{
	Array<float> A, dA, B, dB;
	float weight_decay;

	EltwiseLinear();
	EltwiseLinear(float _weight_decay);

	void init(Shape _in_shape);
	void release() {}
	float* forward(float* input);
	float* backward(float* d_output);
	void save(std::ofstream& file);
	void load(std::ifstream& file);
};
