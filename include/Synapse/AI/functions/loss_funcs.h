#pragma once

#include "Synapse/linear/tensor.h"
#include <string>
#include <map>

namespace syn
{
	syn::Tensor MSE(const syn::Tensor &predicted, const syn::Tensor &desired);
	syn::Tensor MSEDeriv(const syn::Tensor &predicted, const syn::Tensor &desired);
}