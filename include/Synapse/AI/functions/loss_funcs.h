#pragma once

#include "Synapse/linear/tensor.h"
#include <string>
#include <map>

namespace syn {
	syn::Tensor MSE(const syn::Tensor& predicted, const syn::Tensor& desired);
	syn::Tensor MSEPrime(const syn::Tensor& predicted, const syn::Tensor& desired);
	
    extern const std::map<std::string, syn::Tensor(*)(const syn::Tensor&, const syn::Tensor&)> lossFuncs;
	extern const std::map<std::string, syn::Tensor(*)(const syn::Tensor&, const syn::Tensor&)> lossPrimes;
}