
#include <iostream>
#include <amknn.h>

void printTensor(Tensor<float> &arr)
{
    for (int y = 0; y < arr.shape[1]; y++)
    {
        std::cout << "[";

        for (int x = 0; x < arr.shape[0]; x++)
        {
            std::cout << arr(x, y);

            if (x < arr.shape[0] - 1)
                std::cout << ", ";
        }

        std::cout << "]\n";
    }
}

int main()
{
    std::cout << "\nHello C++ !!!\n";

    float arr_data[] =
    {
        1, 2, 3, 4, 5,
        4, 3, 2, 1, 5,
        5, 6, 7, 8, 5,
        1, 1, 2, 2, 5,
        1, 1, 2, 2, 5
    };

    float mask_data[] =
    {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };

    Tensor<float> arr({5, 5}, arr_data);
    Tensor<float> mask({5, 5}, mask_data);

    std::cout << "\nArray : \n";
    printTensor(arr);
    std::cout << "\nMask : \n";
    printTensor(mask);

    arr.copyFrom(mask, { 3, 3 }, { 0, 0, 0, 0 }, { 1, 1, 0, 0 });

    std::cout << "\nResult : \n";
    printTensor(arr);

    return 0;
}
