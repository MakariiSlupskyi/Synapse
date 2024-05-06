#pragma once

#include "Synapse/AI/functions/loss_funcs.h"
#include <functional>
#include <string>
#include <map>

namespace syn
{
    extern const std::map<
        std::string,
        std::function<syn::Tensor(const syn::Tensor &, const syn::Tensor &)>>
        lossFuncs;

    extern const std::map<
        std::string,
        std::function<syn::Tensor(const syn::Tensor &, const syn::Tensor &)>>
        lossDerivs;
}