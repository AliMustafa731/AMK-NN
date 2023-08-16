
#include <fstream>
#include "neural_network.h"

//----------------------------------------------
//  Mean Squared Error Loss Function
//----------------------------------------------
float MSELoss(NeuralNetwork* network, DataSet* data, DataSet* labels)
{
    float total_loss = 0;

    for (int i = 0; i < data->samples_num; i++)
    {
        float* output = network->forward(data->ptr[i]);
        for (int j = 0; j < labels->sample_size; j++)
        {
            float _loss = output[j] - labels->data[j + i * labels->sample_size];
            total_loss += _loss * _loss;
        }
    }

    return (total_loss * 0.5f) / (float)data->samples_num;
}

float* MSELoss(NeuralNetwork* network, float* labels, int batch_size)
{
    NeuralLayer* output_layer = network->output_layer();

    for (int i = 0; i < output_layer->out_size; i++)
    {
        network->loss_gradients[i] = (output_layer->Y[i] - labels[i]) / (float)batch_size;
    }

    return network->loss_gradients.data();
}

void NeuralNetwork::init(Shape _in_shape, std::vector<NeuralLayer*> _layers, Optimizer* _optimizer)
{
    release();

    in_shape = _in_shape;

    layers.reserve(_layers.size());

    for (int i = 0; i < _layers.size(); i++)
    {
        add(_layers[i]);
    }

    set_optimizer(_optimizer);
}

void NeuralNetwork::add(NeuralLayer* layer)
{
    if (layers.size() > 0)
    {
        NeuralLayer* prev = layers[layers.size() - 1];
        layers.push_back(layer);
        layer->in_shape = prev->out_shape;
    }
    else
    {
        layers.push_back(layer);
        layer->in_shape = this->in_shape;
    }
    layer->in_size = layer->in_shape.size();
    layer->init(layer->in_shape);
    layer->allocate(layer->in_shape.size(), layer->out_shape.size());

    loss_gradients.resize(layer->out_size);

    if (layer->parameters.size() > 0)
    {
        for (int j = 0; j < layer->parameters.size(); j++)
        {
            parameters.push_back(&layer->parameters[j]);
        }
    }
}

float* NeuralNetwork::forward(float* input)
{
    float* X = input;

    for (int i = 0; i < layers.size(); i++)
    {
        X = layers[i]->forward(X);
    }

    return output_layer()->Y.data;
}

float* NeuralNetwork::backward(float* d_output)
{
    float* dY = d_output;

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        dY = layers[i]->backward(dY);
    }

    return layers[0]->dX.data;
}

void NeuralNetwork::set_trainable(bool option)
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
            layers[i]->deallocate();
            delete layers[i];
        }
    }
    layers.clear();
    parameters.clear();
    loss_gradients.clear();
}

const char* AMK_FORMAT = "AMKNN";
const int AMK_FORMAT_SIZE = 6;

void NeuralNetwork::save(std::ofstream& file)
{
    int layers_count = layers.size();
    file.write(AMK_FORMAT, AMK_FORMAT_SIZE);
    file.write((char*)&layers_count, sizeof(int));
    file.write((char*)&in_shape, sizeof(Shape));
    file.write((char*)&optimizer->type, sizeof(int));
    optimizer->save(file);

    for (int i = 0; i < layers.size(); i++)
    {
        file.write((char*)&layers[i]->type, sizeof(int));
        file.write((char*)&layers[i]->in_size, sizeof(int));
        file.write((char*)&layers[i]->out_size, sizeof(int));
        file.write((char*)&layers[i]->in_shape, sizeof(Shape));
        file.write((char*)&layers[i]->out_shape, sizeof(Shape));
        layers[i]->save(file);
        layers[i]->save_parameters(file);
    }
}

void NeuralNetwork::load(std::ifstream& file)
{
    release();

    int layers_count, optimizer_type;
    char text[AMK_FORMAT_SIZE];
    file.read(text, AMK_FORMAT_SIZE);
    file.read((char*)&layers_count, sizeof(int));
    file.read((char*)&in_shape, sizeof(Shape));
    file.read((char*)&optimizer_type, sizeof(int));

    if (optimizer_type == GRADIENT_DESCENT) optimizer = new GradientDescent();
    if (optimizer_type == RMS_PROPAGATION) optimizer = new RMSPropagation();
    if (optimizer_type == ADAM) optimizer = new Adam();
    optimizer->load(file);

    if (strcmp(text, AMK_FORMAT))
    {
        return;
    }

    layers.resize(layers_count);
    int in_size, out_size, type;
    Shape _in_shape, _out_shape;

    for (int i = 0; i < layers_count; i++)
    {
        file.read((char*)&type, sizeof(int));
        file.read((char*)&in_size, sizeof(int));
        file.read((char*)&out_size, sizeof(int));
        file.read((char*)&_in_shape, sizeof(Shape));
        file.read((char*)&_out_shape, sizeof(Shape));

        if (type == FULL_LAYER) layers[i] = new FullLayer();
        if (type == CONV_LAYER) layers[i] = new ConvLayer();
        if (type == CONV_TRANSPOSE_LAYER) layers[i] = new ConvTLayer();
        if (type == MAX_POOL_LAYER) layers[i] = new MaxPoolLayer();
        if (type == AVG_POOL_LAYER) layers[i] = new AvgPoolLayer();
        if (type == RElU_LAYER) layers[i] = new RelULayer();
        if (type == RELU_LEAK_LAYER) layers[i] = new RelULeakLayer();
        if (type == SIGMOID_LAYER) layers[i] = new SigmoidLayer();
        if (type == TANH_LAYER) layers[i] = new TanhLayer();
        if (type == SINE_LAYER) layers[i] = new SineLayer();
        if (type == DROPOUT_LAYER) layers[i] = new DropoutLayer();

        layers[i]->load(file);

        layers[i]->type = type;
        layers[i]->in_size = in_size;
        layers[i]->out_size = out_size;
        layers[i]->in_shape = _in_shape;
        layers[i]->out_shape = _out_shape;
        layers[i]->init(layers[i]->in_shape);
        layers[i]->allocate(in_size, out_size);

        layers[i]->load_parameters(file);

        if (layers[i]->parameters.size() > 0)
        {
            for (int j = 0; j < layers[i]->parameters.size(); j++)
            {
                parameters.push_back(&layers[i]->parameters[j]);
            }
        }
    }

    loss_gradients.resize(output_layer()->out_size);
}

void NeuralNetwork::save(std::string filename)
{
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (file.fail()) return;

    save(file);

    file.close();
}

void NeuralNetwork::load(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file.fail()) return;

    load(file);

    file.close();
}

void NeuralNetwork::set_optimizer(Optimizer* _optimizer)
{
    optimizer = _optimizer;
}

