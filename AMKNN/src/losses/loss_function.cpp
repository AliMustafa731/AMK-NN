
#include "loss_function.h"

/// initialize the gradients Tensor
/// @param _grad_size : size of the gradients Tensor
void LossFunction::init(int _grad_size)
{
    gradients.init(_grad_size);
}

/// release the memory allocated by "gradients"
void LossFunction::release()
{
    gradients.release();
}
