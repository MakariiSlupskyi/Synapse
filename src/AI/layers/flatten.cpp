#include "Synapse/AI/layers/flatten.h"
#include "Synapse/linear.h"

void syn::Flatten::save(std::ofstream &file) const
{
	file << "Flatten" << std::endl;
}

syn::Flatten *syn::Flatten::clone() const
{
	return new syn::Flatten();
}

syn::Tensor syn::Flatten::forward(const syn::Tensor &input)
{
	inputShape = input.getShape();
	return syn::reshape(input, {int(input.getData().size()), 1});
}

syn::Tensor syn::Flatten::backward(const syn::Tensor &outputsGrad)
{
	auto t = syn::reshape(outputsGrad, inputShape);
	return t;
}