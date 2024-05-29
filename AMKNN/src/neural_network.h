#pragma once

#include <fstream>
#include <string>

#include <layers/base_layer.h>
#include <data/array.h>

/*
 * A Sequential Neural Netowrk Structure
 * that can cascade many types of layers
 */
struct NeuralNetwork : BaseLayer
{
    /// Array that contains the layers of the network
    Array<BaseLayer*> layers;

    /// @brief construct a neural network with a given input shape
    /// @param _in_shape : input shape
    NeuralNetwork(Shape _in_shape)
    {
        init(_in_shape);
    }
    NeuralNetwork()
    {

    }

    /// @brief Get the output (last) layer
    /// @return output layer (last layer)
    inline BaseLayer* output_layer() { return layers[layers.size() - 1]; }

    /// @brief Get the input (first) layer
    /// @return input layer (first layer)
    inline BaseLayer* input_layer() { return layers[0]; }

    /// @brief construct a neural network with a given input shape
    /// @param _in_shape : input shape
    void init(Shape _in_shape);

    /// @brief Add a layer to the end of the network
    /// @param layer : pointer of the added layer
    void add(BaseLayer* layer);

    /*
     * Forward Pass :
     * forward propagates the input from the first layer to the last
     *
     * @param input
     * The Tensor containing the input (X),
     * which is fed to the first layer
     *
     * @return
     * The output Tensor (Y), which is last layer's output
     */
    Tensor<float>& forward(Tensor<float>& input);

    /*
     * Backward Pass :
     * backward propagates the gradients from the last layer to the first
     *
     * @param output_grad
     * The Tensor containing the Gradients of the
     * objective function with respect to the output (dE/dY)
     *
     * @return
     * The Tensor (dY/dX) of the input layer, containing the gradients of the objective
     * function with respect to first layer's input (X)
     */
    Tensor<float>& backward(Tensor<float>& output_grad);

    /*
     * Toggle the network to accumulate the gradients during the backward pass
     * 
     * if the neural network is toggled "Trainable", then
     * it will accumulate the gradients of the objective
     * with respect to it's parameters after ecah call to "backward",
     * 
     * else, no gradients are accumulated
     *
     * @param state
     * The required state to be applied to all layers
     */
    void setTrainable(bool state);

    /// @brief release the memory allocated by the layers of the network
    void release();

    /// @brief save the whole network (along with it's parameters) into a binary file (custom format)
    /// @param filename : The name of the file to save the network into
    /// @return : true on succes, false on failure
    bool save(std::string filename);

    /// @brief load the newtork (along with it's parameters) from a binary file (custom format)
    /// @param filename :  The name of the file to load the network from
    /// @return : true on succes, false on failure
    bool load(std::string filename);

    /// @brief save the whole network (along with it's parameters) into a binary file (custom format)
    /// @param file : a handle to a previously openned file
    void save(std::ofstream& file);

    /// @brief : load the newtork (along with it's parameters) from a binary file (custom format)
    /// @param file : a handle to a previously openned file
    void load(std::ifstream& file);
};
