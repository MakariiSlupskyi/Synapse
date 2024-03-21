#pragma once

#include "Synapse/linear/tensor.h"

namespace syn {
    syn::Tensor Vector(int size = 1);
    syn::Tensor Vector(int size, const std::vector<double>& data);
    syn::Tensor Vector(const std::vector<double>& data);
}