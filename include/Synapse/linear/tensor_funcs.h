#pragma once

#include "Synapse/linear/tensor.h"
#include <vector>

namespace syn {
    syn::Tensor abs(const syn::Tensor& tensor);
	syn::Tensor square(const syn::Tensor& tensor);
	syn::Tensor log(const syn::Tensor& tensor);
	syn::Tensor exp(const syn::Tensor& tensor);
	syn::Tensor reverse(const syn::Tensor& tensor);
}