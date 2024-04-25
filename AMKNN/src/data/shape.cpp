
#include "shape.h"
#include <cassert>
#include <common.h>

Shape::Shape(){}

Shape::Shape(int a, int b, int c, int d)
{
    dim[0] = a;
    dim[1] = b;
    dim[2] = c;
    dim[3] = d;
}

int Shape::size() const { return dim[0] * dim[1] * dim[2] * dim[3]; }

int Shape::operator[](int idx) const
{
    AMK_ASSERT(idx < 4);
    return dim[idx];
}

int & Shape::operator[](int idx)
{
    AMK_ASSERT(idx < 4);
    return dim[idx];
}
