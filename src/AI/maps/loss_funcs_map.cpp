#include "Synapse/AI/maps/loss_funcs_map.h"

const std::map<std::string, std::function<syn::Tensor(const syn::Tensor &, const syn::Tensor &)>>
    syn::lossFuncs = {
        {"MSE", syn::MSE}};

const std::map<std::string, std::function<syn::Tensor(const syn::Tensor &, const syn::Tensor &)>>
    syn::lossDerivs = {
        {"MSE", syn::MSEDeriv}};