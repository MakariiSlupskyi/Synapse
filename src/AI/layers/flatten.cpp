#include "Synapse/AI/layers/flatten.h"
#include "Synapse/linear/tensor.h"

syn::Tensor syn::Flatten::forward(const syn::Tensor& input) {
	inputShape = input.getShape();
	return input.getReshaped({int(input.getData().size()), 0});
}

syn::Tensor syn::Flatten::backward(const syn::Tensor& outputsGrad) {
	return outputsGrad.getReshaped(inputShape);
}

void syn::Flatten::write(std::ofstream& file) const {
	file << "Flatten" << std::endl;
}