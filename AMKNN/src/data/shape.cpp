
#include "shape.h"
#include <cassert>
#include <common.h>

/*
 * A Structure used to hold Multi-Dimensional shape information
 */

// constructors
Shape::Shape(){}

Shape::Shape(size_t a, size_t b, size_t c, size_t d)
{
    dim[0] = a;
    dim[1] = b;
    dim[2] = c;
    dim[3] = d;
}

/*
 * @return size of the Multi-Dimensional shape,
 * which is just the multipication of all it's dimensions.
 */
size_t Shape::size() const
{
    return dim[0] * dim[1] * dim[2] * dim[3];
}

/// setters and getters for the variable "dim"
size_t Shape::operator[](size_t idx) const
{
    AMK_ASSERT(idx < 4);
    return dim[idx];
}

size_t & Shape::operator[](size_t idx)
{
    AMK_ASSERT(idx < 4);
    return dim[idx];
}
