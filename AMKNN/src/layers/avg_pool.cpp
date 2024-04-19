
#include <layers/avg_pool.h>
#include <utils/convolution.h>

void AvgPoolLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    out_shape[0] = (in_shape[0] - window[0]) / stride[0] + 1;
    out_shape[1] = (in_shape[1] - window[1]) / stride[1] + 1;
    out_shape[2] = in_shape[2];

    in_size = in_shape.size();
    out_size = out_shape.size();

    window[2] = 1;

    NeuralLayer::allocate(in_size, out_size);
}

AvgPoolLayer::AvgPoolLayer()
{
    setTrainable(true);
    type = AVG_POOL_LAYER;
}
AvgPoolLayer::AvgPoolLayer(Shape _window, Shape _stride)
{
    setTrainable(true);
    type = AVG_POOL_LAYER;
    window = _window;
    stride = _stride;
}

Tensor<float>& AvgPoolLayer::forward(Tensor<float>& input)
{
    X = input;

    int dim_x = in_shape[0]*in_shape[1];
    int dim_y = out_shape[0]*out_shape[1];

    float win_size = (float)window.size();

    for (int channel = 0; channel < in_shape[2]; channel++)
    {
        for (int x = 0; x < out_shape[0]; x++)
        {
            for (int y = 0; y < out_shape[1]; y++)
            {
                int start_x = x * stride[0];
                int start_y = y * stride[1];
                int end_x = start_x + window[0];
                int end_y = start_y + window[1];

                float result = 0;

                for (int h = start_x; h < end_x; h++)
                {
                    for (int k = start_y; k < end_y; k++)
                    {
                        result += X[h + k*in_shape[0] + channel*dim_x];
                    }
                }

                Y[x + y*out_shape[0] + channel*dim_y] = result / win_size;
            }
        }
    }

    return Y;
}

Tensor<float>& AvgPoolLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    int dim_x = in_shape[0]*in_shape[1];
    int dim_y = out_shape[0]*out_shape[1];

    float win_size = (float)window.size();

    dX.fill(0);

    for (int channel = 0; channel < in_shape[2]; channel++)
    {
        for (int x = 0; x < out_shape[0]; x++)
        {
            for (int y = 0; y < out_shape[1]; y++)
            {
                int start_x = x * stride[0];
                int start_y = y * stride[1];
                int end_x = start_x + window[0];
                int end_y = start_y + window[1];

                for (int h = start_x; h < end_x; h++)
                {
                    for (int k = start_y; k < end_y; k++)
                    {
                        dX[h + k*in_shape[0] + channel*dim_x] += dY[x + y*out_shape[0] + channel*dim_y] / win_size;
                    }
                }
            }
        }
    }

    return dX;
}

void AvgPoolLayer::save(std::ofstream& file)
{
    NeuralLayer::save(file);
    file.write((char*)&window, sizeof(Shape));
    file.write((char*)&stride, sizeof(Shape));
}

void AvgPoolLayer::load(std::ifstream& file)
{
    NeuralLayer::load(file);
    file.read((char*)&window, sizeof(Shape));
    file.read((char*)&stride, sizeof(Shape));
}
