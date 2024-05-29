#pragma once

#include "loss_function.h"

/*
 * Mean Squared Error (MSE) Loss Function
 */
struct MSELoss : LossFunction
{
    MSELoss() {};

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
    float evaluate(NeuralNetwork& network, Tensor<float>& data, Tensor<float>& labels);

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
    Tensor<float>& gradient(Tensor<float>& prediction, Tensor<float>& ground_truth, float batch_size);
};
