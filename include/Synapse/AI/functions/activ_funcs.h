#pragma once

#include "Synapse/linear/tensor.h"
#include <string>
#include <map>

namespace syn
{
	syn::Tensor relu(const syn::Tensor &tensor);
	syn::Tensor reluDeriv(const syn::Tensor &tensor);

	syn::Tensor leakyRelu(const syn::Tensor &tensor);
	syn::Tensor leakyReluDeriv(const syn::Tensor &tensor);

	syn::Tensor sigmoid(const syn::Tensor &tensor);
	syn::Tensor sigmoidDeriv(const syn::Tensor &tensor);
}