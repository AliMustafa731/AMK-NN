
#include <cfloat>
#include <layers/max_pool.h>
#include <utils/convolution.h>

void MaxPoolLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    out_shape[0] = (in_shape[0] - window[0]) / stride[0] + 1;
    out_shape[1] = (in_shape[1] - window[1]) / stride[1] + 1;
    out_shape[2] = in_shape[2];

    in_size = in_shape.size();
    out_size = out_shape.size();

    max_indices.init(out_size);
    window[2] = 1;

    BaseLayer::allocate(in_size, out_size);
}

MaxPoolLayer::MaxPoolLayer()
{
    setTrainable(true);
    type = MAX_POOL_LAYER;
}
MaxPoolLayer::MaxPoolLayer(Shape _window, Shape _stride)
{
    setTrainable(true);
    type = MAX_POOL_LAYER;
    window = _window;
    stride = _stride;
}

Tensor<float>& MaxPoolLayer::forward(Tensor<float>& input)
{
    X = input;

    int dim_x = in_shape[0]*in_shape[1];
    int dim_y = out_shape[0]*out_shape[1];

    for (int channel = 0 ; channel < in_shape[2] ; channel++)
    {
        for (int x = 0; x < out_shape[0]; x++)
        {
            for (int y = 0; y < out_shape[1]; y++)
            {
                float max = -FLT_MAX;
                int start_x = x * stride[0];
                int start_y = y * stride[1];
                int end_x = start_x + window[0];
                int end_y = start_y + window[1];

                uint64_t max_idx = 0, out_idx, in_idx;

                for (int h = start_x; h < end_x; h++)
                {
                    for (int k = start_y; k < end_y; k++)
                    {
                        in_idx = h + k*in_shape[0] + channel*dim_x;

                        if (X[in_idx] > max)
                        {
                            max = X[in_idx];
                            max_idx = in_idx;
                        }
                    }
                }

                out_idx = x + y*out_shape[0] + channel*dim_y;
                Y[out_idx] = max;
                max_indices[out_idx] = max_idx;
            }
        }
    }

    return Y;
}

Tensor<float>& MaxPoolLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    dX.fill(0);

    for (int i = 0; i < out_size; i++)
    {
        dX[max_indices[i]] += dY[i];
    }

    return dX;
}

void MaxPoolLayer::release()
{
    max_indices.release();
    BaseLayer::release();
}

void MaxPoolLayer::save(std::ofstream& file)
{
    BaseLayer::save(file);
    file.write((char*)&window, sizeof(Shape));
    file.write((char*)&stride, sizeof(Shape));
}

void MaxPoolLayer::load(std::ifstream& file)
{
    BaseLayer::load(file);
    file.read((char*)&window, sizeof(Shape));
    file.read((char*)&stride, sizeof(Shape));
}
