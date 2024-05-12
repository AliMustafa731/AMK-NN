#pragma once

#include <data/tensor.h>
#include <cstdint>

void reverse_bytes(uint8_t* dest, uint8_t* src, int size);

bool load_mnist_images(const char* filename, Tensor<float> &dest, int samples = 0);

bool load_mnist_labels(const char* filename, Tensor<float> &dest, int num_class, int samples = 0);
