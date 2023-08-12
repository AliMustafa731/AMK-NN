
#include "neural_network.h"
#include <cmath>

void NeuralLayer::allocate(int _in_size, int _out_size)
{
	in_size = _in_size;
	out_size = _out_size;
	Y.init(out_size);
	dY.size = out_size;
	X.size = in_size;
	dX.init(in_size);

	for (int j = 0; j < out_size; j++)
	{
		Y[j] = 0;
	}
	for (int j = 0; j < in_size; j++)
	{
		dX[j] = 0;
	}
}

void NeuralLayer::deallocate()
{
	Y.release();
	dX.release();

	if (parameters.size() > 0)
	{
		for (int i = 0; i < parameters.size(); i++)
		{
			parameters[i].release();
		}
	}

	parameters.clear();
}

void NeuralLayer::save_parameters(std::ofstream& file)
{
	int params_num = parameters.size();

	file.write((char*)&trainable, sizeof(bool));
	file.write((char*)&params_num, sizeof(int));

	if (parameters.size() < 1)
	{
		return;
	}

	for (int i = 0; i < parameters.size(); i++)
	{
		file.write((char*)&parameters[i].size, sizeof(int));
		file.write((char*)&parameters[i].decay_rate, sizeof(float));
		file.write((char*)&parameters[i].is_trainable, sizeof(bool));
		file.write((char*)parameters[i].values.data, parameters[i].size * sizeof(float));
		file.write((char*)parameters[i].velocities.data, parameters[i].size * sizeof(float));
		file.write((char*)parameters[i].squared_gradients.data, parameters[i].size * sizeof(float));
	}
}

void NeuralLayer::load_parameters(std::ifstream& file)
{
	int params_num = 0;

	file.read((char*)&trainable, sizeof(bool));
	file.read((char*)&params_num, sizeof(int));

	if (params_num < 1)
	{
		return;
	}

	for (int i = 0; i < parameters.size(); i++)
	{
		int size = 0;
		file.read((char*)&size, sizeof(int));
		file.read((char*)&parameters[i].decay_rate, sizeof(float));
		file.read((char*)&parameters[i].is_trainable, sizeof(bool));
		file.read((char*)parameters[i].values.data, parameters[i].size * sizeof(float));
		file.read((char*)parameters[i].velocities.data, parameters[i].size * sizeof(float));
		file.read((char*)parameters[i].squared_gradients.data, parameters[i].size * sizeof(float));
	}
}

void NeuralLayer::setTrainable(bool state)
{
	trainable = state;

	for (int i = 0; i < parameters.size(); i++)
	{
		parameters[i].is_trainable = state;
	}
}

void FullLayer::init(Shape _in_shape)
{
	parameters.resize(2);
	parameters[0] = Parameter(in_size * out_size, weight_decay);
	parameters[1] = Parameter(out_size);

	W = parameters[0].values;
	dW = parameters[0].gradients;
	B = parameters[1].values;
	dB = parameters[1].gradients;

	for (int i = 0; i < out_size; i++)
	{
		B[i] = random(-1.0f, 1.0f);
		dB[i] = 0;
	}
	for (int i = 0; i < in_size * out_size; i++)
	{
		W[i] = random(-1.0f, 1.0f);
		dW[i] = 0;
	}

	setTrainable(true);
}

float* FullLayer::forward(float* input)
{
	X.data = input;

	for (int i = 0; i < out_size; i++)
	{
		Y[i] = B[i];
		for (int j = 0; j < in_size; j++)
		{
			Y[i] += X[j] * W[i + j * out_size];
		}
	}

	return Y.data;
}

float* FullLayer::backward(float* d_output)
{
	dY.data = d_output;

	if (trainable)
	{
		for (int i = 0; i < out_size; i++)
		{
			dB[i] += dY[i];
			for (int j = 0; j < in_size; j++)
			{
				dW[i + j * out_size] += X[j] * dY[i];
			}
		}
	}

	for (int i = 0; i < in_size; i++)
	{
		dX[i] = 0;
		for (int j = 0; j < out_size; j++)
		{
			dX[i] += W[j + i * out_size] * dY[j];
		}
	}

	return dX.data;
}

void FullLayer::save(std::ofstream& file)
{
	file.write((char*)&weight_decay, sizeof(float));
}

void FullLayer::load(std::ifstream& file)
{
	file.read((char*)&weight_decay, sizeof(float));
}

void ConvLayer::init(Shape _in_shape)
{
	in_shape = _in_shape;
	in_size = in_shape.size();
	out_shape.w = (in_shape.w + 2*padd.w - kernel.w) / stride.w + 1;
	out_shape.h = (in_shape.h + 2*padd.h - kernel.h) / stride.h + 1;
	out_shape.d = kernel.d;
	out_size = out_shape.size();

	parameters.resize(2);
	parameters[0] = Parameter(kernel.size() * in_shape.d, weight_decay);
	parameters[1] = Parameter(out_size);

	K = parameters[0].values;
	dK = parameters[0].gradients;
	B = parameters[1].values;
	dB = parameters[1].gradients;

	for (int i = 0; i < out_size; i++)
	{
		B[i] = random(-1.0f, 1.0f);
		dB[i] = 0;
	}
	for (int j = 0; j < kernel.size() * in_shape.d; j++)
	{
		K[j] = random(-1.0f, 1.0f);
		dK[j] = 0;
	}

	_X = Matrix(in_shape.w, in_shape.h, NULL);
	_dX = Matrix(in_shape.w, in_shape.h, NULL);
	_K = Matrix(kernel.w, kernel.h, NULL);
	_dK = Matrix(kernel.w, kernel.h, NULL);
	_Y = Matrix(out_shape.w, out_shape.h, NULL);
	_dY = Matrix(out_shape.w, out_shape.h, NULL);

	_X_padd.init( in_shape.w + 2*padd.w, in_shape.h + 2*padd.h );
	_dX_padd.init( in_shape.w + 2*padd.w, in_shape.h + 2*padd.h );

	setTrainable(true);
}

float* ConvLayer::forward(float* input)
{
	X.data = input;

	int channels_in = in_shape.d;
	int channels_out = out_shape.d;

	copy(Y, B);

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;
	int dim_k[] = { kernel.w*kernel.h, kernel.w*kernel.h*kernel.d };

	for (int ch_out = 0; ch_out < channels_out; ch_out++)
	{
		_Y.data = Y.data + ch_out*dim_y;
		_K.data = K.data + ch_out*dim_k[0];
		_X.data = X.data;

		for (int ch_in = 0; ch_in < channels_in; ch_in++)
		{
			copy_matrix(_X_padd, _X, { padd.w, padd.h, 0, 0 }, _X.get_rect());
			convolution(_X_padd, _K, _Y, stride);

			_X.data += dim_x;
			_K.data += dim_k[1];
		}
	}

	return Y.data;
}

float* ConvLayer::backward(float* d_output)
{
	dY.data = d_output;

	int channels_in = in_shape.d;
	int channels_out = out_shape.d;
	Shape padd_out(kernel.w - 1, kernel.h - 1, 0);

	if (trainable)
	{
		copy_add(dB, dY);
	}
	fill_array(dX, 0);

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;
	int dim_k[] = { kernel.w*kernel.h, kernel.w*kernel.h*kernel.d };

	for (int ch_out = 0; ch_out < channels_out; ch_out++)
	{
		_Y.data = Y.data + ch_out * dim_y;
		_dY.data = dY.data + ch_out * dim_y;
		_K.data = K.data + ch_out * dim_k[0];
		_dK.data = dK.data + ch_out * dim_k[0];
		_X.data = X.data;
		_dX.data = dX.data;

		for (int ch_in = 0; ch_in < channels_in; ch_in++)
		{
			copy_matrix(_X_padd, _X, { padd.w, padd.h, 0, 0 }, _X.get_rect());

			if (trainable)
			{
				convolution_ex(_X_padd, _dY, _dK, stride);
			}
			convolution_transpose(_dY, _K, _dX_padd);

			copy_matrix(_dX, _dX_padd, { 0, 0, 0, 0 }, { padd.w, padd.h, _dX.w, _dX.h });

			_K.data += dim_k[1];
			_dK.data += dim_k[1];
			_X.data += dim_x;
			_dX.data += dim_x;
		}
	}

	return dX.data;
}

void ConvLayer::save(std::ofstream& file)
{
	file.write((char*)&kernel, sizeof(Shape));
	file.write((char*)&padd, sizeof(Shape));
	file.write((char*)&stride, sizeof(Shape));
	file.write((char*)&weight_decay, sizeof(float));
}

void ConvLayer::load(std::ifstream& file)
{
	file.read((char*)&kernel, sizeof(Shape));
	file.read((char*)&padd, sizeof(Shape));
	file.read((char*)&stride, sizeof(Shape));
	file.read((char*)&weight_decay, sizeof(float));
}

void ConvTLayer::init(Shape _in_shape)
{
	in_shape = _in_shape;
	in_size = in_shape.size();
	out_shape.w = (in_shape.w - 1) * stride.w + kernel.w - 2 * padd.w;
	out_shape.h = (in_shape.h - 1) * stride.h + kernel.h - 2 * padd.h;
	out_shape.d = kernel.d;
	out_size = out_shape.size();

	parameters.resize(2);
	parameters[0] = Parameter(kernel.size() * in_shape.d, weight_decay);
	parameters[1] = Parameter(out_size);

	K = parameters[0].values;
	dK = parameters[0].gradients;
	B = parameters[1].values;
	dB = parameters[1].gradients;

	for (int i = 0; i < out_size; i++)
	{
		B[i] = random(-1.0f, 1.0f);
		dB[i] = 0;
	}
	for (int j = 0; j < kernel.size() * in_shape.d; j++)
	{
		K[j] = random(-1.0f, 1.0f);
		dK[j] = 0;
	}

	_X = Matrix(in_shape.w, in_shape.h, NULL);
	_dX = Matrix(in_shape.w, in_shape.h, NULL);
	_K = Matrix(kernel.w, kernel.h, NULL);
	_dK = Matrix(kernel.w, kernel.h, NULL);
	_Y = Matrix(out_shape.w, out_shape.h, NULL);
	_dY = Matrix(out_shape.w, out_shape.h, NULL);

	_Y_padd.init(out_shape.w + 2 * padd.w, out_shape.h + 2 * padd.h);

	setTrainable(true);
}

float* ConvTLayer::forward(float* input)
{
	X.data = input;

	int channels_in = in_shape.d;
	int channels_out = out_shape.d;

	copy(Y, B);

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;
	int dim_k[] = { kernel.w*kernel.h, kernel.w*kernel.h*kernel.d };

	for (int ch_out = 0; ch_out < channels_out; ch_out++)
	{
		_Y.data = Y.data + ch_out * dim_y;
		_K.data = K.data + ch_out * dim_k[0];
		_X.data = X.data;

		for (int ch_in = 0; ch_in < channels_in; ch_in++)
		{
			convolution_transpose(_X, _K, _Y_padd, stride);
			copy_matrix(_Y, _Y_padd, { 0, 0, 0, 0 }, { padd.w, padd.h, _Y.w, _Y.h });

			_X.data += dim_x;
			_K.data += dim_k[1];
		}
	}

	return Y.data;
}

float* ConvTLayer::backward(float* d_output)
{
	dY.data = d_output;

	int channels_in = in_shape.d;
	int channels_out = out_shape.d;
	Shape padd_out(kernel.w - 1, kernel.h - 1, 0);

	if (trainable)
	{
		copy_add(dB, dY);
	}
	fill_array(dX, 0);

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;
	int dim_k[] = { kernel.w*kernel.h, kernel.w*kernel.h*kernel.d };

	for (int ch_out = 0; ch_out < channels_out; ch_out++)
	{
		_Y.data = Y.data + ch_out * dim_y;
		_dY.data = dY.data + ch_out * dim_y;
		_K.data = K.data + ch_out * dim_k[0];
		_dK.data = dK.data + ch_out * dim_k[0];
		_X.data = X.data;
		_dX.data = dX.data;

		for (int ch_in = 0; ch_in < channels_in; ch_in++)
		{
			if (trainable)
			{
				convolution(_dY, _X, _dK);
			}
			convolution(_dY, _K, _dX);

			_K.data += dim_k[1];
			_dK.data += dim_k[1];
			_X.data += dim_x;
			_dX.data += dim_x;
		}
	}

	return dX.data;
}

void ConvTLayer::save(std::ofstream& file)
{
	file.write((char*)&kernel, sizeof(Shape));
	file.write((char*)&padd, sizeof(Shape));
	file.write((char*)&stride, sizeof(Shape));
	file.write((char*)&weight_decay, sizeof(float));
}

void ConvTLayer::load(std::ifstream& file)
{
	file.read((char*)&kernel, sizeof(Shape));
	file.read((char*)&padd, sizeof(Shape));
	file.read((char*)&stride, sizeof(Shape));
	file.read((char*)&weight_decay, sizeof(float));
}

void MaxPoolLayer::init(Shape _in_shape)
{
	in_shape = _in_shape;
	out_shape.w = (in_shape.w - window.w) / stride.w + 1;
	out_shape.h = (in_shape.h - window.h) / stride.h + 1;
	out_shape.d = in_shape.d;

	in_size = in_shape.size();
	out_size = out_shape.size();

	max_indices = new int[out_size];
	window.d = 1;
}

float* MaxPoolLayer::forward(float* input)
{
	X.data = input;

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;

	for (int channel = 0 ; channel < in_shape.d ; channel++)
	{
		for (int x = 0; x < out_shape.w; x++)
		{
			for (int y = 0; y < out_shape.h; y++)
			{
				float max = -FLT_MAX;
				int start_x = x * stride.w;
				int start_y = y * stride.h;
				int end_x = start_x + window.w;
				int end_y = start_y + window.h;

				uint64_t max_idx = 0, out_idx, in_idx;

				for (int h = start_x; h < end_x; h++)
				{
					for (int k = start_y; k < end_y; k++)
					{
						in_idx = h + k*in_shape.w + channel*dim_x;

						if (X[in_idx] > max)
						{
							max = X[in_idx];
							max_idx = in_idx;
						}
					}
				}

				out_idx = x + y*out_shape.w + channel*dim_y;
				Y[out_idx] = max;
				max_indices[out_idx] = max_idx;
			}
		}
	}

	return Y.data;
}

float* MaxPoolLayer::backward(float* d_output)
{
	dY.data = d_output;

	fill_array(dX, 0);

	for (int i = 0; i < out_size; i++)
	{
		dX[max_indices[i]] += dY[i];
	}

	return dX.data;
}

void MaxPoolLayer::save(std::ofstream& file)
{
	file.write((char*)&window, sizeof(Shape));
	file.write((char*)&stride, sizeof(Shape));
}

void MaxPoolLayer::load(std::ifstream& file)
{
	file.read((char*)&window, sizeof(Shape));
	file.read((char*)&stride, sizeof(Shape));
}

void AvgPoolLayer::init(Shape _in_shape)
{
	in_shape = _in_shape;
	out_shape.w = (in_shape.w - window.w) / stride.w + 1;
	out_shape.h = (in_shape.h - window.h) / stride.h + 1;
	out_shape.d = in_shape.d;

	in_size = in_shape.size();
	out_size = out_shape.size();

	window.d = 1;
}

float* AvgPoolLayer::forward(float* input)
{
	X.data = input;

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;

	float win_size = (float)window.size();

	for (int channel = 0; channel < in_shape.d; channel++)
	{
		for (int x = 0; x < out_shape.w; x++)
		{
			for (int y = 0; y < out_shape.h; y++)
			{
				int start_x = x * stride.w;
				int start_y = y * stride.h;
				int end_x = start_x + window.w;
				int end_y = start_y + window.h;

				float result = 0;

				for (int h = start_x; h < end_x; h++)
				{
					for (int k = start_y; k < end_y; k++)
					{
						result += X[h + k*in_shape.w + channel*dim_x];
					}
				}

				Y[x + y*out_shape.w + channel*dim_y] = result / win_size;
			}
		}
	}

	return Y.data;
}

float* AvgPoolLayer::backward(float* d_output)
{
	dY.data = d_output;

	int dim_x = in_shape.w*in_shape.h;
	int dim_y = out_shape.w*out_shape.h;

	float win_size = (float)window.size();

	fill_array(dX, 0);

	for (int channel = 0; channel < in_shape.d; channel++)
	{
		for (int x = 0; x < out_shape.w; x++)
		{
			for (int y = 0; y < out_shape.h; y++)
			{
				int start_x = x * stride.w;
				int start_y = y * stride.h;
				int end_x = start_x + window.w;
				int end_y = start_y + window.h;

				for (int h = start_x; h < end_x; h++)
				{
					for (int k = start_y; k < end_y; k++)
					{
						dX[h + k*in_shape.w + channel*dim_x] += dY[x + y*out_shape.w + channel*dim_y] / win_size;
					}
				}
			}
		}
	}

	return dX.data;
}

void AvgPoolLayer::save(std::ofstream& file)
{
	file.write((char*)&window, sizeof(Shape));
	file.write((char*)&stride, sizeof(Shape));
}

void AvgPoolLayer::load(std::ifstream& file)
{
	file.read((char*)&window, sizeof(Shape));
	file.read((char*)&stride, sizeof(Shape));
}
