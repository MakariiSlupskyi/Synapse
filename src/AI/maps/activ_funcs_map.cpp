#include "Synapse/AI/maps/activ_funcs_map.h"

const std::map<std::string, std::function<syn::Tensor(const syn::Tensor &)>>
    syn::activFuncs = {
        {"relu", syn::relu},
        {"leaky relu", syn::leakyRelu},
        {"sigmoid", syn::sigmoid}};

const std::map<std::string, std::function<syn::Tensor(const syn::Tensor &)>>
    syn::activDerivs = {
        {"relu", syn::reluDeriv},
        {"leaky relu", syn::leakyReluDeriv},
        {"sigmoid", syn::sigmoidDeriv}};