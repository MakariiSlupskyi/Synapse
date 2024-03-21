#pragma once

#include "Synapse/linear/tensor.h"
#include <string>
#include <map>

namespace syn {
	syn::Tensor relu(const syn::Tensor& tensor);
	syn::Tensor reluPrime(const syn::Tensor& tensor);

	syn::Tensor leakyRelu(const syn::Tensor& tensor);
	syn::Tensor leakyReluPrime(const syn::Tensor& tensor);

	syn::Tensor sigmoid(const syn::Tensor& tensor);
	syn::Tensor sigmoidPrime(const syn::Tensor& tensor);
	
    extern const std::map<std::string, syn::Tensor(*)(const syn::Tensor&)> activFuncs;
    extern const std::map<std::string, syn::Tensor(*)(const syn::Tensor&)> activPrimes;
}