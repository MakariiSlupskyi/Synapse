#pragma once

#include "Synapse/linear/tensor.h"
#include <string>
#include <vector>

namespace syn {
	syn::Tensor abs(const syn::Tensor& tensor);
	syn::Tensor square(const syn::Tensor& tensor);
	syn::Tensor log(const syn::Tensor& tensor);
	syn::Tensor exp(const syn::Tensor& tensor);
	syn::Tensor reverse(const syn::Tensor& tensor);

	syn::Tensor correlate2d(const syn::Tensor& input, const syn::Tensor& kenrel, const std::string& type);
	syn::Tensor convolve2d(const syn::Tensor& input, const syn::Tensor& kenrel, const std::string& type);

	std::vector<std::string> split(std::string str, char separator=' ');

	std::vector<int> splitToI(std::string str, char separator=' ');
	std::vector<double> splitToD(std::string str, char separator=' ');
}