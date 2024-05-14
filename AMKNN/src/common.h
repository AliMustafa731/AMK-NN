#pragma once

// optimizers definitions
enum OptimizerType
{
    GRADIENT_DESCENT,
    RMS_PROPAGATION,
    ADAM
};

// layers definitions
enum LayerType
{
    FULL_LAYER,
    CONV_LAYER,
    CONV_TRANSPOSE_LAYER,
    MAX_POOL_LAYER,
    AVG_POOL_LAYER,
    RElU_LAYER,
    SIGMOID_LAYER,
    TANH_LAYER,
    RELU_LEAK_LAYER,
    DROPOUT_LAYER,
    SINE_LAYER,
    ELTWISE_LINEAR_LAYER
};

#include <cassert>

#define AMK_ASSERT(x) assert(x)

#ifndef NULL
#define NULL 0
#endif
