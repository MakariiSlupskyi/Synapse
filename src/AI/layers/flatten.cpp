#include "Synapse/AI/layers/flatten.h"
#include "Synapse/linear.h"

syn::Tensor syn::Flatten::forward(const syn::Tensor& input) {
	inputShape = input.getShape();
	return syn::reshape(input, {int(input.getData().size()), 1});
}

#include <iostream>

syn::Tensor syn::Flatten::backward(const syn::Tensor& outputsGrad) {
	auto t = syn::reshape(outputsGrad, inputShape);
	return t;
}

void syn::Flatten::write(std::ofstream& file) const {
	file << "Flatten" << std::endl;
}