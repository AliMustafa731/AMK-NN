
#include "activations/activation_layers.h"
#include "utils/utils.h"

float* RelULayer::forward(float* input)
{
	X.data = input;

	for (int i = 0; i < X.size; i++)
	{
		if (X[i] > 0)
		{
			Y[i] = X[i];
		}
		else
		{
			Y[i] = 0;
		}
	}

	return Y.data;
}

float* RelULayer::backward(float* d_output)
{
	dY.data = d_output;

	for (int i = 0; i < X.size; i++)
	{
		if (X[i] > 0)
		{
			dX[i] = dY[i];
		}
		else
		{
			dX[i] = 0;
		}
	}

	return dX.data;
}

float* RelULeakLayer::forward(float* input)
{
	X.data = input;

	for (int i = 0; i < X.size; i++)
	{
		if (X[i] > 0)
		{
			Y[i] = X[i];
		}
		else
		{
			Y[i] = alpha * X[i];
		}
	}

	return Y.data;
}

float* RelULeakLayer::backward(float* d_output)
{
	dY.data = d_output;

	for (int i = 0; i < X.size; i++)
	{
		if (X[i] > 0)
		{
			dX[i] = dY[i];
		}
		else
		{
			dX[i] = alpha * dY[i];
		}
	}

	return dX.data;
}

void RelULeakLayer::save(std::ofstream& file)
{
	file.write((char*)&alpha, sizeof(float));
}

void RelULeakLayer::load(std::ifstream& file)
{
	file.read((char*)&alpha, sizeof(float));
}

float* SigmoidLayer::forward(float* input)
{
	X.data = input;
	for (int i = 0; i < out_size; i++)
	{
		Y[i] = sigmoid(X[i]);
	}
	return Y.data;
}

float* SigmoidLayer::backward(float* d_output)
{
	for (int i = 0; i < in_size; i++)
	{
		dX[i] = d_sigmoid_optimized(Y[i]) * d_output[i];
	}
	return dX.data;
}

float* TanhLayer::forward(float* input)
{
	X.data = input;
	for (int i = 0; i < out_size; i++)
	{
		Y[i] = tanH(X[i]);
	}
	return Y.data;
}

float* TanhLayer::backward(float* d_output)
{
	for (int i = 0; i < in_size; i++)
	{
		dX[i] = d_tanH_optimized(Y[i]) * d_output[i];
	}
	return dX.data;
}

float* SquaredLayer::forward(float* input)
{
	X.data = input;

	for (int i = 0; i < X.size; i++)
	{
		if (X[i] > 1.0f || X[i] < -1.0f)
		{
			Y[i] = 1.0f;
		}
		else
		{
			Y[i] = X[i] * X[i];
		}
	}

	return Y.data;
}

float* SquaredLayer::backward(float* d_output)
{
	dY.data = d_output;

	for (int i = 0; i < X.size; i++)
	{
		if (X[i] >= -1.0f && X[i] <= 1.0f)
		{
			dX[i] = 2.0f * X[i] * dY[i];
		}
		else
		{
			dX[i] = 0;
		}
	}

	return dX.data;
}

void DropoutLayer::init(Shape _in_shape)
{
	out_size = in_size;
	out_shape = in_shape;

	mask.init(in_size);
}

float* DropoutLayer::forward(float* input)
{
	X.data = input;

	for (int i = 0; i < X.size; i++)
	{
		if (trainable)
		{
			if (random() < P)
			{
				mask[i] = 1.0f;
			}
			else
			{
				mask[i] = 0;
			}

			Y[i] = X[i] * mask[i] / P;
		}
		else
		{
			Y[i] = X[i] / P;
		}
	}

	return Y.data;
}

float* DropoutLayer::backward(float* d_output)
{
	dY.data = d_output;

	for (int i = 0; i < X.size; i++)
	{
		if (trainable)
		{
			dX[i] = dY[i] * mask[i] / P;
		}
		else
		{
			dX[i] = dY[i] / P;
		}
	}

	return dX.data;
}

void DropoutLayer::save(std::ofstream& file)
{
	file.write((char*)&P, sizeof(float));
}

void DropoutLayer::load(std::ifstream& file)
{
	file.read((char*)&P, sizeof(float));
}
