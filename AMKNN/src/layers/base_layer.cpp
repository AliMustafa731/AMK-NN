
#include <layers/base_layer.h>
#include <layers/neural_layers.h>
#include <activations/activation_layers.h>
#include <cmath>

// allocate "Input & Output" Tensors
void BaseLayer::allocate(size_t _in_size, size_t _out_size)
{
    in_size = _in_size;
    out_size = _out_size;

    Y.init(out_size);
    dX.init(in_size);

    setTrainable(true);
}

void BaseLayer::deallocate()
{
    Y.release();
    dX.release();

    for (int i = 0; i < parameters.size(); i++)
    {
        parameters[i].release();
    }

    parameters.release();
}

// release the memory allocated by the "Input & Output & Learned Parameters"
void BaseLayer::release()
{
    deallocate();
}

// save the layer along with the "Learned Parameters" to a file
void BaseLayer::save(std::ofstream& file)
{
    // save layer info
    file.write((char*)&type, sizeof(int));
    file.write((char*)&in_size, sizeof(int));
    file.write((char*)&out_size, sizeof(int));
    file.write((char*)&in_shape, sizeof(Shape));
    file.write((char*)&out_shape, sizeof(Shape));

    // save parameters
    int params_num = parameters.size();

    file.write((char*)&trainable, sizeof(bool));
    file.write((char*)&params_num, sizeof(int));

    for (int i = 0; i < parameters.size(); i++)
    {
        parameters[i].save(file);
    }
}

// load the layer along with the "Learned Parameters" from a file
void BaseLayer::load(std::ifstream& file)
{
    release();

    // load layer info
    file.read((char*)&type, sizeof(int));
    file.read((char*)&in_size, sizeof(int));
    file.read((char*)&out_size, sizeof(int));
    file.read((char*)&in_shape, sizeof(Shape));
    file.read((char*)&out_shape, sizeof(Shape));

    init(in_shape);

    // load parameters
    int params_num = 0;

    file.read((char*)&trainable, sizeof(bool));
    file.read((char*)&params_num, sizeof(int));

    for (int i = 0; i < parameters.size(); i++)
    {
        parameters[i].load(file);
    }
}

// construct a new layer from a file along with "Learned Parameters"
// choose the suitable data type based on the flag "type",
// to allow for runtime polymorphisim.
BaseLayer* BaseLayer::loadFromFile(std::ifstream & file)
{
    BaseLayer* layer = NULL;
    int type;

    file.read((char*)&type, sizeof(int));

    if (type == FULL_LAYER) layer = new FullLayer();
    if (type == CONV_LAYER) layer = new ConvLayer();
    if (type == CONV_TRANSPOSE_LAYER) layer = new ConvTLayer();
    if (type == MAX_POOL_LAYER) layer = new MaxPoolLayer();
    if (type == AVG_POOL_LAYER) layer = new AvgPoolLayer();
    if (type == RElU_LAYER) layer = new RelULayer();
    if (type == RELU_LEAK_LAYER) layer = new RelULeakLayer();
    if (type == SIGMOID_LAYER) layer = new SigmoidLayer();
    if (type == TANH_LAYER) layer = new TanhLayer();
    if (type == SINE_LAYER) layer = new SineLayer();
    if (type == DROPOUT_LAYER) layer = new DropoutLayer();
    if (type == ELTWISE_LINEAR_LAYER) layer = new EltwiseLinear();

    file.seekg((int)file.tellg() - sizeof(int));

    layer->load(file);

    return layer;
}

// set the state of the "Learned Parameters"
// if (TRUE), then "Gradients" are accumulated
// in each call to "backward".
void BaseLayer::setTrainable(bool state)
{
    trainable = state;

    for (int i = 0; i < parameters.size(); i++)
    {
        parameters[i].is_trainable = state;
    }
}

