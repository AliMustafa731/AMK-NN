#pragma once

#include "layers/base_layer.h"

//----------------------------------------------
//  Transposed Convolutional Layer
//----------------------------------------------
struct ConvTLayer : NeuralLayer
{
	Array<float> K, dK, B, dB;
	Matrix _X, _dX, _Y, _Y_padd, _dY, _K, _dK;
	Shape kernel, padd, stride;
	float weight_decay;

	ConvTLayer();
	ConvTLayer(Shape _kernel, Shape _stride = Shape(1, 1, 0), Shape _padd = Shape(0, 0, 0), float _weight_decay = 0);

	void init(Shape _in_shape);
	void release() {}
	float* forward(float* input);
	float* backward(float* d_output);
	void save(std::ofstream& file);
	void load(std::ifstream& file);
};
