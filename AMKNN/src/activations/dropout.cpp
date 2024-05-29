
#include <activations/dropout.h>
#include <utils/random.h>

/*
 * "Dropout" layer randomly drops (set to zero)
 * some of it's inputs with a propability "P"
 * to reduce overfitting
 */

/// @brief setup (input/output) shapes
/// @param _in_shape : input shape
void DropoutLayer::init(Shape _in_shape)
{
    in_shape = _in_shape;
    in_size = in_shape.size();
    out_shape = in_shape;
    out_size = in_size;

    mask.init(in_size);

    BaseLayer::allocate(in_size, out_size);
}

/// constructors
DropoutLayer::DropoutLayer(float _P)
{
    P = _P;
    type = DROPOUT_LAYER;
}
DropoutLayer::DropoutLayer()
{
    P = 0;
    type = DROPOUT_LAYER;
}

/// @brief release the memory of "mask"
void DropoutLayer::release()
{
    mask.release();
    BaseLayer::release();
}

/*
 * Forward Pass :
 * drop some inputs randomly depending on the probability "P",
 * store the result in the output
 *
 * @param input : input Tensor
 * @return : output Tensor
 */
Tensor<float>& DropoutLayer::forward(Tensor<float>& input)
{
    X = input;

    for (int i = 0; i < X.size(); i++)
    {
        if (trainable)
        {
            if (random() < P)
            {
                mask[i] = 1.0f;
            }
            else
            {
                mask[i] = 0;
            }

            Y[i] = X[i] * mask[i] / P;
        }
        else
        {
            Y[i] = X[i] / P;
        }
    }

    return Y;
}

/*
 * Backward Pass :
 * caculate the gradients of the objective with respect to the "Input" and "Learned Parameters"
 * only masked inputs (multiplied by 1) recieve the gradients
 *
 * @param output_grad :
 * Tensor containing gradients of the objective with respect to the output Tensor
 * 
 * @return :
 * Tensor containing gradients with respect to "Input"
 */
Tensor<float>& DropoutLayer::backward(Tensor<float>& output_grad)
{
    dY = output_grad;

    for (int i = 0; i < X.size(); i++)
    {
        if (trainable)
        {
            dX[i] = dY[i] * mask[i] / P;
        }
        else
        {
            dX[i] = dY[i] / P;
        }
    }

    return dX;
}

/// @brief save "P" hyper parameter to a file
/// @param file : handle to a previously openned file
void DropoutLayer::save(std::ofstream& file)
{
    BaseLayer::save(file);
    file.write((char*)&P, sizeof(float));
}

/// @brief load "P" hyper parameter from a file
/// @param file : handle to a previously openned file
void DropoutLayer::load(std::ifstream& file)
{
    BaseLayer::load(file);
    file.read((char*)&P, sizeof(float));
}
