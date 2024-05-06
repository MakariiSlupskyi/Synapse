#include "Synapse/AI/dataset.h"

#include <cstdlib>
#include <stdexcept>

syn::Dataset::Dataset(const syn::Data &inputs, const syn::Data &outputs)
    : inputs(inputs), outputs(outputs)
{
}

void syn::Dataset::setInputs(const syn::Data &other)
{
    if (inputs.size() != other.size())
    {
        throw std::runtime_error("Invalid data given for setting input data");
    }
    inputs = other;
}

void syn::Dataset::setOutputs(const syn::Data &other)
{
    if (outputs.size() != other.size())
    {
        throw std::runtime_error("Invalid data given for setting output data");
    }
    outputs = other;
}

syn::Dataset &syn::Dataset::shuffle()
{
    int seed = std::rand();
    inputs.shuffle(seed);
    outputs.shuffle(seed);
    return *this;
}

syn::Dataset syn::Dataset::merge(const syn::Dataset &other)
{
    return syn::Dataset(inputs.merge(other.getInputs()), outputs.merge(other.getOutputs()));
}

syn::Dataset syn::Dataset::extract(int start, int size) const
{
    return syn::Dataset(inputs.extract(start, size), outputs.extract(start, size));
}