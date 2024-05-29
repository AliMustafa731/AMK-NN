#pragma once

#include <data/tensor.h>

struct NeuralNetwork;

/*
 * Loss Function (a.k.a. Objective function)
 * the goal of training is to minimize this funtion
 */
struct LossFunction
{
    Tensor<float> gradients;

    LossFunction(){}
    ~LossFunction(){}

    /// initialize the gradients Tensor
    /// @param _grad_size : size of the gradients Tensor
    void init(int _grad_size);

    /// release the memory allocated by "gradients"
    void release();

    /*
     * Calculate the loss for a given dataset and neural network
     *
     * @param network : the nueral network to calculate the loss on
     * 
     * @param data : the dataset to calculate the loss on
     * 
     * @param labels : the ground truth of the dataset
     * 
     * @return a single number representing the loss
     */
    virtual float evaluate(NeuralNetwork& network, Tensor<float>& data, Tensor<float>& labels) = 0;

    /*
     * Calculate the gradients of the loss function, and back propagates it
     * 
     * @param prediction : the predicted output (e.g. from a neural network)
     * 
     * @param ground_truth : the expected output
     * 
     * @param batch_size : size of the mini-batch (gradients are divied by this number)
     * 
     * @return The Tensor containing the gradients of the loss
     */
    virtual Tensor<float>& gradient(Tensor<float>& prediction, Tensor<float>& ground_truth, float batch_size) = 0;
};
