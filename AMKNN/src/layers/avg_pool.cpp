
#include "layers/neural_layers.h"
#include "utils/utils.h"

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

AvgPoolLayer::AvgPoolLayer()
{
    type = AVG_POOL_LAYER;
}
AvgPoolLayer::AvgPoolLayer(Shape _window, Shape _stride)
{
    type = AVG_POOL_LAYER;
    window = _window;
    stride = _stride;
}

