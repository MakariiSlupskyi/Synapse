#pragma

#include "Synapse/AI/optimizers/GD.h"
#include "Synapse/AI/optimizers/SGD.h"

namespace syn {
    extern const std::map<
        std::string, syn::Optimizer* (*)(syn::Model* model, std::vector<syn::Layer*>* layers)
    > optimizers;
}