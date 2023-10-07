
#include <optimizers/adam.h>

//----------------------------------
//   Optimize The Parameters
//----------------------------------
void Adam::update(List<Parameter*> &parameters)
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
            float& squared_grad = p->squared_gradients[j];

            velocity = beta1 * velocity + (1.0f - beta1) * gradient;
            squared_grad = beta2 * squared_grad + (1.0f - beta2) * gradient*gradient;

            float velocity_corr = velocity / (1.0f - beta1 + 1e-8f);
            float squared_corr = squared_grad / (1.0f - beta2 + 1e-8f);

            value -= learning_rate * velocity_corr / (sqrt(squared_corr) + 1e-8f);
            value -= p->decay_rate * value;
            gradient = 0;
        }
    }
}

Adam::Adam()
{
    type = ADAM;
}
Adam::Adam(float _learning_rate, float _beta1, float _beta2)
{
    type = ADAM;
    learning_rate = _learning_rate;
    beta1 = _beta1;
    beta2 = _beta2;
}

void Adam::save(std::ofstream& file)
{
    file.write((char*)&learning_rate, sizeof(float));
    file.write((char*)&beta1, sizeof(float));
    file.write((char*)&beta2, sizeof(float));
}
void Adam::load(std::ifstream& file)
{
    file.read((char*)&learning_rate, sizeof(float));
    file.read((char*)&beta1, sizeof(float));
    file.read((char*)&beta2, sizeof(float));
}
