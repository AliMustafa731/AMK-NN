
#include <optimizers/gradient_descent.h>

//----------------------------------
//   Optimize The Parameters
//----------------------------------
void GradientDescent::update(List<Parameter*> &parameters)
{
    for (auto n = parameters.base; n != NULL; n = n->next)
    {
        Parameter *p = n->value;

        if (p->is_trainable == false) continue;

        for (int j = 0; j < p->size(); j++)
        {
            float& value = p->values[j];
            float& gradient = p->gradients[j];
            float& velocity = p->velocities[j];

            velocity = momentum * velocity + (1.0f - momentum) * gradient;

            value -= learning_rate * velocity;
            value -= p->decay_rate * value;
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
