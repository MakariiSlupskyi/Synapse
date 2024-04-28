#include "Synapse/AI/layers/activation.h"
#include "Synapse/AI/functions.h"

syn::Activation::Activation(const std::string& type)
    : type(type), activFunc(syn::activFuncs.at(type)), activPrime(syn::activPrimes.at(type))
{}

syn::Activation::Activation(std::ifstream& file) {
	std::string type;
	std::getline(file, type);
	*this = syn::Activation(type);
}

syn::Tensor syn::Activation::forward(const syn::Tensor& inputs) {
    this->inputs = inputs;
    return activFunc(inputs);
}

syn::Tensor syn::Activation::backward(const syn::Tensor& outGrad) {
    return outGrad * activPrime(inputs);
}

void syn::Activation::write(std::ofstream& file) const {
	file << "Activation\n";
	file << type << std::endl;
}