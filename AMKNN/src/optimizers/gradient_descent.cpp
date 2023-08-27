
#include "optimizers/gradient_descent.h"

void GradientDescent::update(List<Parameter*> &parameters)
{
    for (auto n = parameters.base; n != NULL; n = n->next)
    {
        Parameter *p = n->value;

        if (!(p->is_trainable))
        {
            continue;
        }

        for (int j = 0; j < p->size; j++)
        {
            float gradient = p->gradients[j];

            p->velocities[j] = momentum * p->velocities[j] + (1.0f - momentum) * gradient;

            p->values[j] -= learning_rate * p->velocities[j];
            p->gradients[j] = 0;

            p->values[j] -= p->decay_rate * p->values[j];
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

void GradientDescent::save(std::ofstream& file)
{
    file.write((char*)&learning_rate, sizeof(float));
    file.write((char*)&momentum, sizeof(float));
}
void GradientDescent::load(std::ifstream& file)
{
    file.read((char*)&learning_rate, sizeof(float));
    file.read((char*)&momentum, sizeof(float));
}
