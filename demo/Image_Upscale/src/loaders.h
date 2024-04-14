#pragma once

#include <data/dataset.h>
#include <cstdint>

void reverse_bytes(uint8_t* dest, uint8_t* src, int size);

bool load_mnist_images(const char* filename, DataSet &dest, int samples = 0);

bool load_mnist_labels(const char* filename, DataSet &dest, int num_class, int samples = 0);
