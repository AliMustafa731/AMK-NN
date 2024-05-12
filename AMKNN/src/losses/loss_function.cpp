
#include "loss_function.h"

void LossFunction::init(int _grad_size) { gradients.init(_grad_size); }
void LossFunction::release() { gradients.release(); }
