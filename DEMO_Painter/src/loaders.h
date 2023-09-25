#pragma once

#include <utils/graphics.h>
#include <data/tensor.h>

int readImage(const char* filename, Tensor<Color> &target);

int readImage(const char* filename, Tensor<Colorf> &target);

void normalize(Tensor<Colorf> &target);

void normalize(Tensor<float> &target);

void denormalize(Tensor<Colorf> &target);

void denormalize(Tensor<float> &target);
