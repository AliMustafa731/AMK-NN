
#include "mse_loss.h"
#include <neural_network.h>

/*
 * Mean Squared Error (MSE) Loss Function
 */

/*
 * Calculate the loss for a given dataset and neural network,
 * loss = 0.5 * (Y - Y`)^2
 *
 * @param network : the nueral network to calculate the loss on
 * 
 * @param data : the dataset to calculate the loss on
 * 
 * @param labels : the ground truth of the dataset
 * 
 * @return a single number representing the loss
 */
float MSELoss::evaluate(NeuralNetwork& network, Tensor<float>& data, Tensor<float>& labels)
{
    float total_loss = 0;

    for (size_t i = 0; i < data.shape[3]; i++)
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

/*
 * Calculate the gradients of the loss function, and back propagates it,
 * gradient = (Y - Y`)
 * 
 * @param prediction : the predicted output (e.g. from a neural network)
 * 
 * @param ground_truth : the expected output
 * 
 * @param batch_size : size of the mini-batch (gradients are divied by this number)
 * 
 * @return The Tensor containing the gradients of the loss
 */
Tensor<float>& MSELoss::gradient(Tensor<float>& prediction, Tensor<float>& ground_truth, float batch_size)
{
    for (int i = 0; i < prediction.size(); i++)
    {
        gradients[i] = (prediction[i] - ground_truth[i]) / (float)batch_size;
    }

    return gradients;
}
