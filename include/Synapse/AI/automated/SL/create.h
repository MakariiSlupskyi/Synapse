#pragma once

#include "Synapse/AI/data.h"
#include "Synapse/AI/model.h"
#include "Synapse/AI/layers.h"

namespace syn {
    syn::Model create(const syn::Data& inputs, const syn::Data& labels);
}