#include "Synapse/linear/tensor_funcs.h"
#include <cmath>
#include <stdexcept>

syn::Tensor syn::abs(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return std::abs(x); });
}

syn::Tensor syn::square(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return x * x; });
}

syn::Tensor syn::log(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return std::log(x); });
}

syn::Tensor syn::exp(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return std::exp(x); });
}

syn::Tensor syn::reverse(const syn::Tensor& tensor) {
    return syn::Tensor(tensor).reverse();
}

syn::Tensor syn::reshape(const syn::Tensor& tensor, const std::vector<int>& shape) {
	return syn::Tensor(tensor).reshape(shape);
}

syn::Tensor syn::correlate2d(const syn::Tensor& input_, const syn::Tensor& kernel, const std::string& type) {
	if (input_.getShape().size() != 2 || kernel.getShape().size() != 2) {
		throw std::invalid_argument("Invalid arguments for getting block of a tensor.");
	}

	// Prepare input tensor relying on type
	syn::Tensor input;
	if (type == "valid") {
		input = input_;
	} else if (type == "full") {
		input.reshape({
			input_.getShape()[0] + 2 * (kernel.getShape()[0] - 1),
			input_.getShape()[1] + 2 * (kernel.getShape()[1] - 1)
		});
		input.setBlock({kernel.getShape()[0] - 1, kernel.getShape()[1] - 1}, input_);
	}
	
	// Calculate result
	syn::Tensor res({input.getShape()[0] - kernel.getShape()[0] + 1, input.getShape()[1] - kernel.getShape()[1] + 1});
	for (int i = 0; i < res.getShape()[0]; ++i) {
		for (int j = 0; j < res.getShape()[1]; ++j) {
			res({i, j}) = (input.block({i, j}, kernel.getShape()) * kernel).sum();
		}
	}

	return res;
}

syn::Tensor syn::convolve2d(const syn::Tensor& input, const syn::Tensor& kernel, const std::string& type) {
	return syn::correlate2d(input, syn::reverse(kernel), type);
}
