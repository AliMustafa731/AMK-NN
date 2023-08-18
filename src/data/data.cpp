
#include "data/data.h"

void DataSet::init(Shape _shape, int _samples_num)
{
    shape = _shape;
    samples_num = _samples_num;
    sample_size = shape.size();

    data.init(samples_num * sample_size);
    make_ptr_list<float>(ptr, data, sample_size);
}

void DataSet::release()
{
    sample_size = 0;
    samples_num = 0;
    data.release();
    ptr.release();
}
