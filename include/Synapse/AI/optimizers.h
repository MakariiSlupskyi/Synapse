#pragma

#include "Synapse/AI/optimizers/GD.h"
#include "Synapse/AI/optimizers/SGD.h"
#include <string>
#include <map>

namespace syn {
    extern const std::map<
        std::string, syn::Optimizer* (*)(syn::Model* model, std::vector<syn::ILayer*>* layers)
    > optimizers;
}