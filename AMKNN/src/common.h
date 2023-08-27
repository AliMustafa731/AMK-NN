#pragma once

// optimizers definitions
#define GRADIENT_DESCENT 0
#define RMS_PROPAGATION 1
#define ADAM 2

// layers definitions
#define FULL_LAYER 0
#define CONV_LAYER 1
#define CONV_TRANSPOSE_LAYER 2
#define MAX_POOL_LAYER 3
#define AVG_POOL_LAYER 4
#define RElU_LAYER 5
#define SIGMOID_LAYER 6
#define TANH_LAYER 7
#define RELU_LEAK_LAYER 8
#define DROPOUT_LAYER 9
#define SINE_LAYER 10
#define ELTWISE_LINEAR_LAYER 11

#define AMK_ASSERT(x) assert(x)

#ifndef NULL
#define NULL 0
#endif
