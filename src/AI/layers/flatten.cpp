#include "Synapse/AI/layers/flatten.h"
#include "Synapse/linear/vector.h"

syn::Tensor syn::Flatten::forward(const syn::Tensor& input) {
	inputShape = input.getShape();
	return syn::Vector(input.getData());
}

syn::Tensor syn::Flatten::backward(const syn::Tensor& outputsGrad) {
	return outputsGrad.reshape(inputShape);
}

void syn::Flatten::write(std::ofstream& file) const {
	file << "Flatten" << std::endl;
}