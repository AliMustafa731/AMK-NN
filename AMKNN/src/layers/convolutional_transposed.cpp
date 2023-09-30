
#include <cstring>
#include <layers/convolutional_transposed.h>
#include <utils/convolution.h>
#include <utils/random.h>

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

    NeuralLayer::allocate(in_size, out_size);
}

ConvTLayer::ConvTLayer()
{
    setTrainable(true);
    type = CONV_TRANSPOSE_LAYER;
}
ConvTLayer::ConvTLayer(Shape _kernel, Shape _stride, Shape _padd, float _weight_decay)
{
    setTrainable(true);
    type = CONV_TRANSPOSE_LAYER;
    kernel = _kernel;
    stride = _stride;
    padd = _padd;
    weight_decay = _weight_decay;
}

Tensor<float>& ConvTLayer::forward(Tensor<float>& input)
{
    X = input;

    int channels_in = in_shape.d;
    int channels_out = out_shape.d;

    std::memcpy(Y.data, B.data, Y.size() * sizeof(float));

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
            Matrix::copy(_Y, _Y_padd, { 0, 0, 0, 0 }, { padd.w, padd.h, _Y.w, _Y.h });

            _X.data += dim_x;
            _K.data += dim_k[1];
        }
    }

    return Y;
}

Tensor<float>& ConvTLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    int channels_in = in_shape.d;
    int channels_out = out_shape.d;
    Shape padd_out(kernel.w - 1, kernel.h - 1, 0);

    if (trainable)
    {
        dB.add(dY);
    }
    dX.fill(0);

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

    return dX;
}

void ConvTLayer::save(std::ofstream& file)
{
    NeuralLayer::save(file);
    file.write((char*)&kernel, sizeof(Shape));
    file.write((char*)&padd, sizeof(Shape));
    file.write((char*)&stride, sizeof(Shape));
    file.write((char*)&weight_decay, sizeof(float));
}

void ConvTLayer::load(std::ifstream& file)
{
    NeuralLayer::load(file);
    file.read((char*)&kernel, sizeof(Shape));
    file.read((char*)&padd, sizeof(Shape));
    file.read((char*)&stride, sizeof(Shape));
    file.read((char*)&weight_decay, sizeof(float));
}
