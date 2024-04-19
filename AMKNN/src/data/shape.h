#pragma once

struct Shape
{
private:
    int dim[4];
public:

    Shape();
    Shape(int a, int b = 1, int c = 1, int d = 1);

    int size();
    int operator[](int idx) const;
    int& operator[](int idx);
};
