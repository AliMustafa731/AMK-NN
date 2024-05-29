
#include <neural_network.h>

/*
 * A Sequential Neural Netowrk Structure
 * that can cascade many types of layers
 */

/// @brief construct a neural network with a given input shape
/// @param _in_shape : input shape
void NeuralNetwork::init(Shape _in_shape)
{
    release();
    in_shape = _in_shape;
}

/// @brief Add a layer to the end of the network
/// @param layer : pointer of the added layer
void NeuralNetwork::add(BaseLayer* layer)
{
    if (layers.size() < 1) // the first layer
    {
        layer->in_shape = this->in_shape;
    }
    else
    {
        BaseLayer* prev = layers[layers.size() - 1];
        layer->in_shape = prev->out_shape;
    }

    layer->init(layer->in_shape);
    layers.add(layer);

    Y = output_layer()->Y;
    dX = input_layer()->dX;

    for (int j = 0; j < layer->parameters.size(); j++)
    {
        parameters.add(layer->parameters[j]);
    }
}

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
Tensor<float>& NeuralNetwork::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < layers.size(); i++)
    {
        X = layers[i]->forward(X);
    }

    return Y;
}

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
Tensor<float>& NeuralNetwork::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        dY = layers[i]->backward(dY);
    }

    return dX;
}

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
void NeuralNetwork::setTrainable(bool state)
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->setTrainable(state);
    }
}

/// @brief release the memory allocated by the layers of the network
void NeuralNetwork::release()
{
    for (int i = 0; i < layers.size(); i++)
    {
        if (layers[i] != NULL)
        {
            layers[i]->release();
            delete layers[i];
        }
    }
    layers.release();
    parameters.release();
}

const char* AMK_FORMAT = "AMKNN";
const int AMK_FORMAT_SIZE = 6;

/// @brief save the whole network (along with it's parameters) into a binary file (custom format)
/// @param filename : The name of the file to save the network into
/// @return : true on succes, false on failure
void NeuralNetwork::save(std::ofstream& file)
{
    int layers_count = layers.size();
    file.write(AMK_FORMAT, AMK_FORMAT_SIZE);
    file.write((char*)&layers_count, sizeof(int));
    file.write((char*)&in_shape, sizeof(Shape));

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->save(file);
    }
}

/// @brief load the newtork (along with it's parameters) from a binary file (custom format)
/// @param filename :  The name of the file to load the network from
/// @return : true on succes, false on failure
void NeuralNetwork::load(std::ifstream& file)
{
    release();

    int layers_count;
    char text[AMK_FORMAT_SIZE];
    file.read(text, AMK_FORMAT_SIZE);
    file.read((char*)&layers_count, sizeof(int));
    file.read((char*)&in_shape, sizeof(Shape));

    if (std::string(text) != std::string(AMK_FORMAT)) return;

    layers.resize(layers_count);

    for (int i = 0; i < layers_count; i++)
    {
        layers[i] = BaseLayer::loadFromFile(file);

        for (int j = 0; j < layers[i]->parameters.size(); j++)
        {
            parameters.add(layers[i]->parameters[j]);
        }
    }
}

/// @brief save the whole network (along with it's parameters) into a binary file (custom format)
/// @param file : a handle to a previously openned file
bool NeuralNetwork::save(std::string filename)
{
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (file.fail()) return false;

    NeuralNetwork::save(file);

    file.close();
    return true;
}

/// @brief : load the newtork (along with it's parameters) from a binary file (custom format)
/// @param file : a handle to a previously openned file
bool NeuralNetwork::load(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file.fail()) return false;

    NeuralNetwork::load(file);

    file.close();
    return true;
}
