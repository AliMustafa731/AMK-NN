
#include "optimizers.h"
#include "neural_network.h"


void Adam::update(std::vector<Parameter*> &parameters)
{
    for (int i = 0; i < parameters.size(); i++)
    {
        Parameter *p = parameters[i];

        if (!(p->is_trainable))
        {
            continue;
        }

        for (int j = 0; j < p->size; j++)
        {
            float gradient = p->gradients[j];

            p->velocities[j] = beta1 * p->velocities[j] + (1.0f - beta1) * gradient;
            p->squared_gradients[j] = beta2 * p->squared_gradients[j] + ((1.0f - beta2) * gradient*gradient);

            float velocity_corr = p->velocities[j] / (1.0f - beta1 + 1e-8f);
            float squared_corr = p->squared_gradients[j] / (1.0f - beta2 + 1e-8f);

            p->values[j] -= learning_rate * velocity_corr / (sqrt(squared_corr) + 1e-8f);
            p->gradients[j] = 0;

            p->values[j] -= p->decay_rate * p->values[j];
        }
    }
}

void GradientDescent::update(std::vector<Parameter*> &parameters)
{
    for (int i = 0; i < parameters.size(); i++)
    {
        Parameter *p = parameters[i];

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

void RMSPropagation::update(std::vector<Parameter*> &parameters)
{
    for (int i = 0; i < parameters.size(); i++)
    {
        Parameter *p = parameters[i];

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
