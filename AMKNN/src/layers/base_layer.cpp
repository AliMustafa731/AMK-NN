
#include "layers/base_layer.h"
#include <cmath>

void NeuralLayer::allocate(int _in_size, int _out_size)
{
    in_size = _in_size;
    out_size = _out_size;

    Y.init(out_size);
    dY.init(out_size, NULL);
    X.init(in_size, NULL);
    dX.init(in_size);

    for (int j = 0; j < out_size; j++)
    {
        Y[j] = 0;
    }
    for (int j = 0; j < in_size; j++)
    {
        dX[j] = 0;
    }
}

void NeuralLayer::deallocate()
{
    Y.release();
    dX.release();
    this->release();

    for (int i = 0; i < parameters.size(); i++)
    {
        parameters[i].release();
    }

    parameters.release();
}

void NeuralLayer::save_parameters(std::ofstream& file)
{
    int params_num = parameters.size();

    file.write((char*)&trainable, sizeof(bool));
    file.write((char*)&params_num, sizeof(int));

    if (parameters.size() < 1)
    {
        return;
    }

    for (int i = 0; i < parameters.size(); i++)
    {
        file.write((char*)&parameters[i].size, sizeof(int));
        file.write((char*)&parameters[i].decay_rate, sizeof(float));
        file.write((char*)&parameters[i].is_trainable, sizeof(bool));
        file.write((char*)parameters[i].values.data, parameters[i].size * sizeof(float));
        file.write((char*)parameters[i].velocities.data, parameters[i].size * sizeof(float));
        file.write((char*)parameters[i].squared_gradients.data, parameters[i].size * sizeof(float));
    }
}

void NeuralLayer::load_parameters(std::ifstream& file)
{
    int params_num = 0;

    file.read((char*)&trainable, sizeof(bool));
    file.read((char*)&params_num, sizeof(int));

    if (params_num < 1)
    {
        return;
    }

    for (int i = 0; i < parameters.size(); i++)
    {
        int size = 0;
        file.read((char*)&size, sizeof(int));
        file.read((char*)&parameters[i].decay_rate, sizeof(float));
        file.read((char*)&parameters[i].is_trainable, sizeof(bool));
        file.read((char*)parameters[i].values.data, parameters[i].size * sizeof(float));
        file.read((char*)parameters[i].velocities.data, parameters[i].size * sizeof(float));
        file.read((char*)parameters[i].squared_gradients.data, parameters[i].size * sizeof(float));
    }
}

void NeuralLayer::setTrainable(bool state)
{
    trainable = state;

    for (int i = 0; i < parameters.size(); i++)
    {
        parameters[i].is_trainable = state;
    }
}

