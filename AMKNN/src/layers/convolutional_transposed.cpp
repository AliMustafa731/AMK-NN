
#include <cstring>
#include <layers/convolutional_transposed.h>
#include <utils/convolution.h>
#include <utils/random.h>

void ConvTLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape[0] = (in_shape[0] - 1) * stride[0] + kernel[0] - 2 * padd[0];
    out_shape[1] = (in_shape[1] - 1) * stride[1] + kernel[1] - 2 * padd[1];
    out_shape[2] = kernel[2];
    out_size = out_shape.size();

    parameters.resize(2);
    parameters[0] = Parameter(kernel.size() * in_shape[2], weight_decay);
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
    for (int j = 0; j < kernel.size() * in_shape[2]; j++)
    {
        K[j] = random(-1.0f, 1.0f);
        dK[j] = 0;
    }

    _X = Matrix(in_shape[0], in_shape[1], NULL);
    _dX = Matrix(in_shape[0], in_shape[1], NULL);
    _K = Matrix(kernel[0], kernel[1], NULL);
    _dK = Matrix(kernel[0], kernel[1], NULL);
    _Y = Matrix(out_shape[0], out_shape[1], NULL);
    _dY = Matrix(out_shape[0], out_shape[1], NULL);

    _Y_padd.init(out_shape[0] + 2 * padd[0], out_shape[1] + 2 * padd[1]);

    setTrainable(true);

    BaseLayer::allocate(in_size, out_size);
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

    int channels_in = in_shape[2];
    int channels_out = out_shape[2];

    std::memcpy(Y.data, B.data, Y.size() * sizeof(float));

    int dim_x = in_shape[0]*in_shape[1];
    int dim_y = out_shape[0]*out_shape[1];
    int dim_k[] = { kernel[0]*kernel[1], kernel[0]*kernel[1]*kernel[2] };

    for (int ch_out = 0; ch_out < channels_out; ch_out++)
    {
        _Y.data = Y.data + ch_out * dim_y;
        _K.data = K.data + ch_out * dim_k[0];
        _X.data = X.data;

        for (int ch_in = 0; ch_in < channels_in; ch_in++)
        {
            convolution_transpose(_X, _K, _Y_padd, stride);
            Matrix::copy(_Y, _Y_padd, { 0, 0, 0, 0 }, { padd[0], padd[1], _Y.w, _Y.h });

            _X.data += dim_x;
            _K.data += dim_k[1];
        }
    }

    return Y;
}

Tensor<float>& ConvTLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    int channels_in = in_shape[2];
    int channels_out = out_shape[2];
    Shape padd_out(kernel[0] - 1, kernel[1] - 1, 0);

    if (trainable)
    {
        dB.add(dY);
    }
    dX.fill(0);

    int dim_x = in_shape[0]*in_shape[1];
    int dim_y = out_shape[0]*out_shape[1];
    int dim_k[] = { kernel[0]*kernel[1], kernel[0]*kernel[1]*kernel[2] };

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
    BaseLayer::save(file);
    file.write((char*)&kernel, sizeof(Shape));
    file.write((char*)&padd, sizeof(Shape));
    file.write((char*)&stride, sizeof(Shape));
    file.write((char*)&weight_decay, sizeof(float));
}

void ConvTLayer::load(std::ifstream& file)
{
    BaseLayer::load(file);
    file.read((char*)&kernel, sizeof(Shape));
    file.read((char*)&padd, sizeof(Shape));
    file.read((char*)&stride, sizeof(Shape));
    file.read((char*)&weight_decay, sizeof(float));
}
