
#include "mse_loss.h"
#include <neural_network.h>

//----------------------------------------------
//  Mean Squared Error Loss Function
//  forward :  loss = 0.5 * (Y - Y`)^2
//  backward : grad_wrt_Y = (Y - Y`)
//----------------------------------------------

// evaluate the loss
float MSELoss::evaluate(NeuralNetwork& network, Tensor<float>& data, Tensor<float>& labels)
{
    float total_loss = 0;

    for (int i = 0; i < data.shape[3]; i++)
    {
        Tensor<float> sample = data.slice({ data.shape[0], data.shape[1] }, { 0, 0, 0, i });
        Tensor<float> label = labels.slice({ labels.shape.size() }, { 0, 0, 0, i });

        Tensor<float>& output = network.forward(sample);

        for (int j = 0; j < output.size(); j++)
        {
            float _loss = output[j] - label[j];
            total_loss += _loss * _loss;
        }
    }

    return (total_loss * 0.5f) / (float)data.shape[3];
}

// calculate the loss gradients
Tensor<float>& MSELoss::gradient(Tensor<float>& output, Tensor<float>& label, float batch_size)
{
    for (int i = 0; i < output.size(); i++)
    {
        gradients[i] = (output[i] - label[i]) / (float)batch_size;
    }

    return gradients;
}
