#pragma once

#include "Synapse/AI/interfaces/layer.h"
#include <functional>
#include <string>
#include <map>

namespace syn
{
    extern const std::map<
        std::string,
        std::function<syn::ILayer *()>>
        layers;
}