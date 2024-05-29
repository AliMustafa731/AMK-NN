#pragma once

#include <data/array.h>
#include <data/tensor.h>
#include <optimizers/parameter.h>
#include <utils/geometry.h>
#include <fstream>

struct BaseLayer
{
    Tensor<float> X, dX, Y, dY;
    Array<Parameter> parameters;
    size_t in_size, out_size, type;
    Shape in_shape, out_shape;
    bool trainable;

    BaseLayer() {}

    // allocate "Input & Output" Tensors
    void allocate(size_t _in_size, size_t _out_size);
    void deallocate();

    //-------Implemented By Derrived Classes-------//

    // Takes "input" Tensor,
    // Use it to caculate the "output" Tensor,
    // And return the "Output" Tensor
    virtual Tensor<float>& forward(Tensor<float>& input) = 0;

    // Takes the "gradients of the objective with respect to the output" Tensor,
    // Use it to caculate the gradients with respect to the "Input" and "Learned Parameters" Tensors,
    // And return the Tensor containing gradients with respect to "Input".
    virtual Tensor<float>& backward(Tensor<float>& output_grad) = 0;

    // setup the shape of the output
    // setup & allocate the "Learned Parameters"
    virtual void init(Shape _in_shape) = 0; // implemented by derrived classes

    // release the memory allocated by the "Input & Output & Learned Parameters"
    virtual void release();

    // set the state of the "Learned Parameters"
    // if (TRUE), then "Gradients" are accumulated
    // in each call to "backward".
    virtual void setTrainable(bool state);

    // save the layer along with the "Learned Parameters" to a file
    virtual void save(std::ofstream& file);

    // load the layer along with the "Learned Parameters" from a file
    virtual void load(std::ifstream& file);

    // construct a new layer from a file along with "Learned Parameters"
    // choose the suitable data type based on the flag "type",
    // to allow for runtime polymorphisim
    static BaseLayer* loadFromFile(std::ifstream& file);
};
