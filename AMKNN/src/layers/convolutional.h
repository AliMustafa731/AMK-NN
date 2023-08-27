#pragma once

#include "layers/base_layer.h"

//----------------------------------------------
//  Convolutional Layer
//----------------------------------------------
struct ConvLayer : NeuralLayer
{
	Array<float> K, dK, B, dB;
	Matrix _X, _X_padd, _dX, _dX_padd, _Y, _dY, _K, _dK;
	Shape kernel, padd, stride;
	float weight_decay;

	ConvLayer();
	ConvLayer(Shape _kernel, Shape _stride = Shape(1, 1, 0), Shape _padd = Shape(0, 0, 0), float _weight_decay = 0);

	void init(Shape _in_shape);
	void release() {}
	float* forward(float* input);
	float* backward(float* d_output);
	void save(std::ofstream& file);
	void load(std::ifstream& file);
};
