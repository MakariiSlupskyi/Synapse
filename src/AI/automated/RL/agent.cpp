#include "Synapse/AI/automated/RL/agent.h"

syn::Agent::Agent(const syn::Model &model) : model(model)
{
}

syn::Tensor syn::Agent::getOutput(const syn::Tensor &inputs)
{
    return model.predict(inputs);
}

void syn::Agent::reward(double value)
{
    score += value;
}

void syn::Agent::forget()
{
    model.randomize();
}

void syn::Agent::mutate(double alpha)
{
    model.tune(alpha);
}