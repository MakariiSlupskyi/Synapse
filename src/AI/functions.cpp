#include "Synapse/AI/functions/activ_funcs.h"
#include "Synapse/AI/functions/loss_funcs.h"
#include "Synapse/AI/functions.h"


// Activation function maps
const std::map<std::string, syn::Tensor(*)(const syn::Tensor&)> syn::activFuncs = {
	{ "relu", syn::relu },
	{ "leaky relu", syn::leakyRelu },
	{ "sigmoid", syn::sigmoid }
};

const std::map<std::string, syn::Tensor(*)(const syn::Tensor&)> syn::activPrimes = {
	{ "relu", syn::reluPrime },
	{ "leaky relu", syn::leakyReluPrime },
	{ "sigmoid", syn::sigmoidPrime }
};


// Loss function maps
const std::map<std::string, syn::Tensor (*)(const syn::Tensor&, const syn::Tensor&)> syn::lossFuncs {
	{ "MSE", syn::MSE }
};

const std::map<std::string, syn::Tensor (*)(const syn::Tensor&, const syn::Tensor&)> syn::lossPrimes {
	{ "MSE", syn::MSEPrime }
};