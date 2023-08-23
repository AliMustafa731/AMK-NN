
#include "optimizers/optimizers.h"

void RMSPropagation::update(List<Parameter*> &parameters)
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

            p->squared_gradients[j] = beta * p->squared_gradients[j] + ((1.0f - beta) * gradient*gradient);

            p->values[j] -= learning_rate * gradient / (sqrt(p->squared_gradients[j]) + 1e-8f);
            p->gradients[j] = 0;

            p->values[j] -= p->decay_rate * p->values[j];
        }
    }
}

RMSPropagation::RMSPropagation()
{
    type = RMS_PROPAGATION;
}
RMSPropagation::RMSPropagation(float _learning_rate, float _beta)
{
    type = RMS_PROPAGATION;
    learning_rate = _learning_rate;
    beta = _beta;
}

void RMSPropagation::save(std::ofstream& file)
{
    file.write((char*)&learning_rate, sizeof(float));
    file.write((char*)&beta, sizeof(float));
}
void RMSPropagation::load(std::ifstream& file)
{
    file.read((char*)&learning_rate, sizeof(float));
    file.read((char*)&beta, sizeof(float));
}
