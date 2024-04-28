#include "Synapse/AI/optimizers/GD.h"
#include "Synapse/AI/optimizers/SGD.h"
#include "Synapse/AI/optimizers.h"

const std::map<
    std::string, syn::Optimizer* (*)(syn::Model* model, std::vector<syn::ILayer*>* layers)
> syn::optimizers = {
    { "GD",  [](syn::Model* model, std::vector<syn::ILayer*>* layers) -> syn::Optimizer* {
        return new syn::GD(model, layers); }},
    {"SGD", [](syn::Model* model, std::vector<syn::ILayer*>* layers) -> syn::Optimizer* {
        return new syn::SGD(model, layers); }}
};