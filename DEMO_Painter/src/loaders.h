#pragma once

#include "utils/graphics.h"
#include "data/buffer.h"

int readImage(const char* filename, Buffer<Color> &target);

int readImage(const char* filename, Buffer<Colorf> &target);

void normalize(Buffer<Colorf> &target);

void denormalize(Buffer<Colorf> &target);
