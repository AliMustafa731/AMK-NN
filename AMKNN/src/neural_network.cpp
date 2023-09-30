
#include <neural_network.h>
#include <string>

//----------------------------------------------
//  Mean Squared Error Loss Function
//  forward :  loss = 0.5 * (Y - Y`)^2
//  backward : d_loss_wrt_Y = (Y - Y`)
//----------------------------------------------

// evaluate the loss
float MSELoss::evaluate(NeuralNetwork& network, DataSet& data, DataSet& labels)
{
    float total_loss = 0;

    for (int i = 0; i < data.samples_num; i++)
    {
        Tensor<float>& output = network.forward(data[i]);
        for (int j = 0 ; j < output.size() ; j++)
        {
            float _loss = output[j] - labels[i][j];
            total_loss += _loss * _loss;
        }
    }

    return (total_loss * 0.5f) / (float)data.samples_num;
}

// calculate the loss gradients
Tensor<float>& MSELoss::gradient(NeuralNetwork& network, Tensor<float>& label, int batch_size)
{
    NeuralLayer* output_layer = network.output_layer();

    for (int i = 0; i < output_layer->out_size; i++)
    {
        gradients[i] = (output_layer->Y[i] - label[i]) / (float)batch_size;
    }

    return gradients;
}

void LossFunction::init(int _grad_size) { gradients.init(_grad_size); }
void LossFunction::release() { gradients.release(); }

//-----------------------------------------
//  Sequential Neural Network Structure
//-----------------------------------------
void NeuralNetwork::init(Shape _in_shape, std::vector<NeuralLayer*> _layers)
{
    release();
    in_shape = _in_shape;
    layers.reserve(_layers.size());

    for (int i = 0; i < _layers.size(); i++)
    {
        NeuralNetwork::add(_layers[i]);
    }
}

void NeuralNetwork::add(NeuralLayer* layer)
{
    if (layers.size() < 1) // the first layer
    {
        layer->in_shape = this->in_shape;
    }
    else
    {
        NeuralLayer* prev = layers[layers.size() - 1];
        layer->in_shape = prev->out_shape;
    }

    layer->init(layer->in_shape);
    layers.add(layer);

    for (int j = 0; j < layer->parameters.size(); j++)
    {
        parameters.add(&layer->parameters[j]);
    }
}

Tensor<float>& NeuralNetwork::forward(Tensor<float>& input)
{
    Tensor<float> X = input;

    for (int i = 0; i < layers.size(); i++)
    {
        X = layers[i]->forward(X);
    }

    return output_layer()->Y;
}

Tensor<float>& NeuralNetwork::backward(Tensor<float>& output_grad)
{
    Tensor<float> dY = output_grad;

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        dY = layers[i]->backward(dY);
    }

    return input_layer()->dX;
}

void NeuralNetwork::setTrainable(bool option)
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->setTrainable(option);
    }
}

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
        layers[i] = NeuralLayer::loadFromFile(file);

        for (int j = 0; j < layers[i]->parameters.size(); j++)
        {
            parameters.add(&layers[i]->parameters[j]);
        }
    }
}

bool NeuralNetwork::save(std::string filename)
{
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (file.fail()) return false;

    NeuralNetwork::save(file);

    file.close();
    return true;
}

bool NeuralNetwork::load(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file.fail()) return false;

    NeuralNetwork::load(file);

    file.close();
    return true;
}
