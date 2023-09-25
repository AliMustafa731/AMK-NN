
#include <layers/max_pool.h>
#include <utils/convolution.h>

void MaxPoolLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    out_shape.w = (in_shape.w - window.w) / stride.w + 1;
    out_shape.h = (in_shape.h - window.h) / stride.h + 1;
    out_shape.d = in_shape.d;

    in_size = in_shape.size();
    out_size = out_shape.size();

    max_indices.init(out_size);
    window.d = 1;
}

Tensor<float>& MaxPoolLayer::forward(Tensor<float>& input)
{
    X = input;

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

MaxPoolLayer::MaxPoolLayer()
{
    type = MAX_POOL_LAYER;
}
MaxPoolLayer::MaxPoolLayer(Shape _window, Shape _stride)
{
    type = MAX_POOL_LAYER;
    window = _window;
    stride = _stride;
}
