#include "Synapse/AI/automated/SL/create.h"
#include "Synapse/AI/layers.h"
#include <stdexcept>
#include <vector>


/// @brief Function, that automatically generates and teaches models by training data
syn::Model syn::create(const syn::Data& inputs, const syn::Data& labels) {
    return syn::Model(syn::ModelBuilder(inputs, labels));
}