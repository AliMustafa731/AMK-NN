#pragma once

#include "utils/graphics.h"
#include "data/array.h"
#include "data/dataset.h"

void reverse_bytes(unsigned char* dest, unsigned char* src, int size);

bool load_mnist_images(const char* filename, DataSet &dest, int samples = 0);

bool load_mnist_labels(const char* filename, DataSet &dest, int num_class, int samples = 0);
