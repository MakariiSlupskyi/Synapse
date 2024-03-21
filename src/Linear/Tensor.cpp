#include "Synapse/linear/tensor.h"

#include <numeric>
#include <functional>
#include <cstdlib>
#include <execution>
#include <algorithm>
#include <stdexcept>


syn::Tensor::Tensor() : shape({1}), data({1}), dataSize(1)
{}

syn::Tensor::Tensor(const std::vector<int>& shape)
    : shape(shape), dataSize(1)
{
    for (int i = 0; i < shape.size(); ++i) { dataSize *= shape[i]; }
    data.resize(dataSize);
}

syn::Tensor::Tensor(const std::vector<int>& shape, const std::vector<double>& data)
    : Tensor(shape)
{
    if (data.size() != dataSize) {
        throw std::runtime_error("Invalid size of given tensor data");
    }
    this->data = data;
}

syn::Tensor& syn::Tensor::fill(double value) {
	std::fill(data.begin(), data.end(), value);
    return *this;
}

syn::Tensor& syn::Tensor::fill(const std::vector<double>& values) {
    if (data.size() != dataSize) {
        throw std::runtime_error("Invalid size of given tensor data");
    }
    for (int i = 0; i < dataSize; ++i) { data[i] = values[i]; }
    return *this;
}


syn::Tensor& syn::Tensor::zeros() {
	std::fill(data.begin(), data.end(), 0.0);
    return *this;
}

syn::Tensor& syn::Tensor::ones() {
	std::fill(data.begin(), data.end(), 1.0);
    return *this;
}

syn::Tensor& syn::Tensor::randomize() {
	for (int i = 0; i < data.size(); ++i) {
		data[i] = std::rand() / (float)RAND_MAX * 2.0 - 1.0;
	}
	return *this;
}

syn::Tensor& syn::Tensor::reshape(const std::vector<int>& shape) {
	this->shape = shape;
    for (int i = 0; i < shape.size(); ++i) { dataSize *= shape[i]; }
    data.resize(dataSize);
	return *this;
}

syn::Tensor& syn::Tensor::reverse() {
	std::reverse(this->data.begin(), this->data.end());
	return *this;
}

double syn::Tensor::sum() const {
	return std::accumulate(data.begin(), data.end(), 0.0);
}

double syn::Tensor::max() const {
	return *std::max_element(data.begin(), data.end());
}

double syn::Tensor::min() const {
	return *std::min_element(data.begin(), data.end());
}

syn::Tensor syn::Tensor::matMul(const syn::Tensor& other) const {
	if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0]) {
		throw std::runtime_error("Invalid tensor shape for matrix multiplication");
	}
	syn::Tensor result({shape[0], other.shape[1]});
	result.fill(0.0);
	for (int i = 0; i < shape[0]; ++i) {
		for (int k = 0; k < shape[1]; ++k) {
			for (int j = 0; j < other.shape[1]; ++j) {
				result({i, j}) += (*this)({i, k}) * other({k, j});
			}
		}
	}
	return result;
}

syn::Tensor syn::Tensor::matTrans() const {
	if (shape.size() != 2) {
		throw std::runtime_error("Invalid tensor shape for matrix transposing");
	}
    const int nRow = shape[0];
    const int nCol = shape[1];
	syn::Tensor result({nCol, nRow});
	for (int i = 0; i < nRow; ++i) {
		for (int j = 0; j < nCol; ++j) {
			result({j, i}) = this->operator()({i, j});
		}
	}
	return result;
}

syn::Tensor& syn::Tensor::apply(double (*func)(double)) {
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(), func);
	return *this;
}

syn::Tensor syn::Tensor::apply(double (*func)(double)) const {
	syn::Tensor res(*this);
	std::transform(std::execution::par, res.data.begin(), res.data.end(), res.data.begin(), func);
	return res;
}

syn::Tensor syn::Tensor::operator+(const syn::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for addition");
	}
	syn::Tensor res(*this);
	std::transform(
		std::execution::par_unseq,
		res.data.begin(), res.data.end(),
		other.data.begin(), res.data.begin(),
		std::plus<double>());
	return res;
}

syn::Tensor& syn::Tensor::square() {
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
		[](double x) -> double { return x * x; });
	return *this;
}

double syn::Tensor::operator()(const std::vector<int>& indices) const {
    return data[calcIndex(indices)];
}

double& syn::Tensor::operator()(const std::vector<int>& indices) {
    return data[calcIndex(indices)];
}

bool syn::Tensor::operator==(const syn::Tensor& other) const {
	if (shape.size() != other.shape.size()) { return false; }
	for (int i = 0; i < shape.size(); ++i) {
		if (shape[i] != other.shape[i]) { return false; }
	}
	return true;
}

bool syn::Tensor::operator!=(const syn::Tensor& other) const {
	return !this->operator==(other);
}

syn::Tensor syn::Tensor::operator-(const syn::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for subtracting");
	}
	syn::Tensor res(*this);
	std::transform(
		std::execution::par_unseq,
		res.data.begin(), res.data.end(),
		other.data.begin(), res.data.begin(),
		std::minus<double>());
	return res;
}

syn::Tensor syn::Tensor::operator*(const syn::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for multiplying");
		}
	syn::Tensor res(*this);
		std::transform(
		std::execution::par_unseq,
		res.data.begin(), res.data.end(),
		other.data.begin(), res.data.begin(),
		std::multiplies<double>());
	return res;
}

syn::Tensor syn::Tensor::operator/(const syn::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for division");
	}
	syn::Tensor res(*this);
	std::transform(
		std::execution::par_unseq,
		res.data.begin(), res.data.end(),
		other.data.begin(), res.data.begin(),
		std::divides<double>());
	return res;
}

syn::Tensor& syn::Tensor::operator+=(const syn::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for addition");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::plus<double>());
	return *this;
}

syn::Tensor& syn::Tensor::operator-=(const syn::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for subtracting");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::minus<double>());
	return *this;
}

syn::Tensor& syn::Tensor::operator*=(const syn::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for multiplying");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::multiplies<double>());
	return *this;
}

syn::Tensor& syn::Tensor::operator/=(const syn::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::runtime_error("Invalid given tensor shape for division");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::divides<double>());
	return *this;
}

syn::Tensor syn::Tensor::operator+(double value) const {
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
        [value](double x) { return x + value; });
	return result;
}

syn::Tensor syn::Tensor::operator-(double value) const {
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
        [value](double x) { return x - value; });
	return result;
}

syn::Tensor syn::Tensor::operator*(double value) const {
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
        [value](double x) { return x * value; });
	return result;
}

syn::Tensor syn::Tensor::operator/(double value) const {
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
        [value](double x) { return x / value; });
	return result;
}

syn::Tensor& syn::Tensor::operator+=(double value) {
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
        [value](double x) { return x + value; });
	return *this;
}

syn::Tensor& syn::Tensor::operator-=(double value) {
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
        [value](double x) { return x - value; });
	return *this;
}

syn::Tensor& syn::Tensor::operator*=(double value) {
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
        [value](double x) { return x * value; });
	return *this;
}

syn::Tensor& syn::Tensor::operator/=(double value) {
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
        [value](double x) { return x / value; });
	return *this;
}

int syn::Tensor::calcIndex(const std::vector<int>& indices) const {
    int result = 0, multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        result += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return result;
}