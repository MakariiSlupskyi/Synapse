#pragma once

#include "Synapse/AI/functions/activ_funcs.h"
#include <functional>
#include <string>
#include <map>

namespace syn
{
    extern const std::map<
        std::string,
        std::function<syn::Tensor(const syn::Tensor &)>>
        activFuncs;

    extern const std::map<
        std::string,
        std::function<syn::Tensor(const syn::Tensor &)>>
        activDerivs;
}