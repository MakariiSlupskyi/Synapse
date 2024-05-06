#pragma once

#include "Synapse/AI/layers/dense.h"
#include "Synapse/AI/layers/activation.h"
#include "Synapse/AI/layers/convolutional.h"
#include "Synapse/AI/layers/pooling.h"
#include "Synapse/AI/layers/flatten.h"
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