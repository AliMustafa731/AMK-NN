
#include "tensor.h"
#include <gui/image.h>

//
// force the compiler to generate code for the following templates :
//
template class Tensor<float>;
template class Tensor<int>;
template class Tensor<uint8_t>;
template class Tensor<Color>;

Tensor<float> _tensor_float_;
Tensor<int> _tensor_int_;
Tensor<uint8_t> _tensor_uint8t_;
Tensor<Color> _tensor_color_;
