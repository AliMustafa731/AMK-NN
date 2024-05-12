
#include <optimizers/rms_propagation.h>

//----------------------------------
//   Optimize The Parameters
//----------------------------------
void RMSPropagation::update(Array<Parameter> &parameters)
{
    for (int i = 0; i < parameters.size(); i++)
    {
        Parameter& p = parameters[i];

        if (p.is_trainable == false) continue;

        for (int j = 0; j < p.size(); j++)
        {
            float& value = p.values[j];
            float& gradient = p.gradients[j];
            float& squared_grad = p.squared_gradients[j];

            squared_grad = beta * squared_grad + (1.0f - beta) * gradient*gradient;

            value -= learning_rate * gradient / (sqrt(squared_grad) + 1e-8f);
            value -= p.decay_rate * value;
            gradient = 0;
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
