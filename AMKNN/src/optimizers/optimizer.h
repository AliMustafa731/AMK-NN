#pragma once

#include <optimizers/parameter.h>
#include <data/array.h>
#include <fstream>

/*
 * Optimizer is an algorithm that's used to optimize (tweak)
 * the parameters of a model to minimize the objective function
 */
struct Optimizer
{
    /// ID used to identify the type of the Optimizer,
    /// used to allow for runtime polymorphism
    int type;

    Optimizer() {}
    virtual ~Optimizer() {}

    /*
     * Update the given parameters (apply an optimization step),
     * then reset the gradients of the parameters to zero.
     * 
     * (implemented by derrived classes)
     * 
     * @param parameters : Array of parameters to be optimized
     */
    virtual void update(Array<Parameter> &parameters) = 0;

    /*
     * save the hyper-parameters of this otimizer to a file,
     *
     * (implemented by derrived classes)
     *
     * @param file : handle to a previously openned file
     */
    virtual void save(std::ofstream& file) = 0;

    /*
     * load the hyper-parameters of this otimizer from a file,
     *
     * (implemented by derrived classes)
     *
     * @param file : handle to a previously openned file
     */
    virtual void load(std::ifstream& file) = 0;
};
