#include "Synapse/AI/layers/activation.h"
#include "Synapse/AI/maps/activ_funcs_map.h"

syn::Activation::Activation(const std::string &type)
	: type(type), function(syn::activFuncs.at(type)), derivative(syn::activDerivs.at(type))
{
}

void syn::Activation::load(std::ifstream &file)
{
	std::string type;
	std::getline(file, type);
	*this = syn::Activation(type);
}

void syn::Activation::save(std::ofstream &file) const
{
	file << "Activation\n";
	file << type << std::endl;
}

syn::Activation *syn::Activation::clone() const
{
	auto res = new syn::Activation(type);
	res->input = input;

	return res;
}

syn::Tensor syn::Activation::forward(const syn::Tensor &inputs)
{
	this->input = input;
	return function(inputs);
}

syn::Tensor syn::Activation::backward(const syn::Tensor &outGrad)
{
	return outGrad * derivative(input);
}
