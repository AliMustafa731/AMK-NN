
#include <optimizers/gradient_descent.h>

//
// The "Gradient Descent" Optimizer
// for more info, see : https://machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure/
//

//
// tweak the parameters & zero thier gradients
//
void GradientDescent::update(Array<Parameter> &parameters)
{
    for (int i = 0 ; i < parameters.size() ; i++)
    {
        Parameter& p = parameters[i];

        if (p.is_trainable == false) continue;

        for (int j = 0; j < p.size(); j++)
        {
            float& value = p.values[j];
            float& gradient = p.gradients[j];
            float& velocity = p.velocities[j];

            velocity = momentum * velocity + (1.0f - momentum) * gradient;

            value -= learning_rate * velocity;
            value -= p.decay_rate * value;
            gradient = 0;
        }
    }
}

GradientDescent::GradientDescent()
{
    type = GRADIENT_DESCENT;
}
GradientDescent::GradientDescent(float _learning_rate, float _momentum)
{
    type = GRADIENT_DESCENT;
    learning_rate = _learning_rate;
    momentum = _momentum;
}

// save hyper parameters to a file
void GradientDescent::save(std::ofstream& file)
{
    file.write((char*)&learning_rate, sizeof(float));
    file.write((char*)&momentum, sizeof(float));
}

// load hyper parameters from a file
void GradientDescent::load(std::ifstream& file)
{
    file.read((char*)&learning_rate, sizeof(float));
    file.read((char*)&momentum, sizeof(float));
}
