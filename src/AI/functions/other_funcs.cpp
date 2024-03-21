#include "Synapse/AI/functions/other_funcs.h"
#include <sstream>
#include <cmath>

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

syn::Tensor syn::correlate2d(const syn::Tensor& input_, const syn::Tensor& kernel, const std::string& type) {
	// if (input_.getShape().size() != 2 || kernel.getShape().size() != 2) {
	// 	throw std::invalid_argument("Invalid arguments for getting block of a tensor.");
	// }

	// syn::Tensor input;
	// if (type == "valid") {
	// 	input = input_;
	// } else if (type == "full") {
	// 	input = input.reshape({
	// 		input_.getShape()[0] + 2 * (kernel.getShape()[0] - 1),
	// 		input_.getShape()[1] + 2 * (kernel.getShape()[1] - 1)
	// 	});
	// 	input.setBlock({kernel.getShape()[0] - 1, kernel.getShape()[1] - 1}, input_);
	// }
	
	// syn::Tensor res({input.getShape()[0] - kernel.getShape()[0] + 1, input.getShape()[1] - kernel.getShape()[1] + 1});
	// for (int i = 0; i < res.getShape()[0]; ++i) {
	// 	for (int j = 0; j < res.getShape()[1]; ++j) {
	// 		res({i, j}) = (input.block({i, j}, kernel.getShape()) * kernel).sum();
	// 	}
	// }

	return kernel;
}

syn::Tensor syn::convolve2d(const syn::Tensor& input, const syn::Tensor& kernel, const std::string& type) {
	return syn::correlate2d(input, syn::reverse(kernel), type);
}

std::vector<std::string> syn::split(std::string str, char separator) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, separator)) {
        result.push_back(item);
    }

    return result;  
}

std::vector<int> syn::splitToI(std::string str, char separator) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, separator)) {
        result.push_back(std::stoi(item));
    }

    return result;  
}

std::vector<double> syn::splitToD(std::string str, char separator) {
    std::vector<double> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, separator)) {
        result.push_back(std::stod(item));
    }

    return result;
}