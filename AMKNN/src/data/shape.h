#pragma once

#include <cstdint>

/*
 * A Structure used to hold Multi-Dimensional shape information
 */
class Shape
{
public:

    Shape();
    Shape(size_t a, size_t b = 1, size_t c = 1, size_t d = 1);

    /*
     * @return size of the Multi-Dimensional shape,
     * which is just the multipication of all it's dimensions.
     */
    size_t size() const;

    /// setters and getters for the variable "dim"
    size_t operator[](size_t idx) const;
    size_t& operator[](size_t idx);

private:

    /// variable used to store the dimensions
    size_t dim[4];
};
