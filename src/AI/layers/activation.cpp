#include "Synapse/AI/layers/activation.h"
#include "Synapse/AI/functions.h"

syn::Activation::Activation(const std::string& type)
    : type(type), func(syn::activFuncs.at(type)), prime(syn::activPrimes.at(type))
{}

syn::Activation::Activation(std::ifstream& file) {
	std::string type;
	std::getline(file, type);
	*this = syn::Activation(type);
}

syn::Tensor syn::Activation::forward(const syn::Tensor& inputs) {
    this->inputs = inputs;
    return func(inputs);
}

syn::Tensor syn::Activation::backward(const syn::Tensor& outGrad) {
    return outGrad * prime(inputs);
}

void syn::Activation::write(std::ofstream& file) const {
	file << "Activation\n";
	file << type << std::endl;
}