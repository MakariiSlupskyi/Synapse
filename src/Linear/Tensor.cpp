#include "Tensor.h"

#include <numeric>
#include <limits>
#include <algorithm>
#include <stdexcept>

ml::Tensor::Tensor() : shape({1}), data({1})
{}

ml::Tensor::Tensor(const std::vector<int>& shape)
    : shape(shape), dataSize(1)
{
    for (int i = 0; i < shape.size(); ++i) { dataSize *= shape[i]; }
    data.resize(dataSize);
}

ml::Tensor::Tensor(const std::vector<int>& shape, const std::vector<double>& data)
    : Tensor(shape)
{
    if (data.size() != dataSize) {
        throw std::runtime_error("Invalid tensor data size!");
    }
    this->data = data;
}

ml::Tensor& ml::Tensor::fill(double value) {
	for (int i = 0; i < dataSize; ++i) { data[i] = value; }
    return *this;
}

ml::Tensor& ml::Tensor::fill(const std::vector<double>& values) {
    if (data.size() != dataSize) {
        throw std::runtime_error("Invalid tensor data size!");
    }
    for (int i = 0; i < dataSize; ++i) { data[i] = values[i]; }
    return *this;
}

ml::Tensor& ml::Tensor::randomize() {	// FIXME
	for (int i = 0; i < dataSize; ++i) { data[i] = 0.0d; }
    return *this;
}

double ml::Tensor::operator()(const std::vector<int>& indices) const {
    return data[calcIndex(indices)];
}

double& ml::Tensor::operator()(const std::vector<int>& indices) {
    return data[calcIndex(indices)];
}

double ml::Tensor::sum() const {
	return std::accumulate(data.begin(), data.end(), 0);
}

double ml::Tensor::max() const {
	return std::accumulate(data.begin(), data.end(), std::numeric_limits<int>::min(),
        [](int a, int b) { return std::max(a, b); });
}

double ml::Tensor::min() const {
	return std::accumulate(data.begin(), data.end(), std::numeric_limits<int>::max(),
        [](int a, int b) { return std::min(a, b); });
}

bool ml::Tensor::operator==(const ml::Tensor& other) const {
	if (shape.size() != other.shape.size()) { return false; }
	for (int i = 0; i < shape.size(); ++i) {
		if (shape[i] != other.shape[i]) { return false; }
	}
	return true;
}

bool ml::Tensor::operator!=(const ml::Tensor& other) const {
	return !this->operator==(other);
}

ml::Tensor ml::Tensor::operator+(const ml::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] += other.data[i]; }
	return res;
}

ml::Tensor ml::Tensor::operator-(const ml::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] -= other.data[i]; }
	return res;
}

ml::Tensor ml::Tensor::operator*(const ml::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
		}
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] *= other.data[i]; }
	return res;
}

ml::Tensor ml::Tensor::operator/(const ml::Tensor& other) const {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] /= other.data[i]; }
	return res;
}

ml::Tensor& ml::Tensor::operator+=(const ml::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	for(int i = 0; i < dataSize; ++i) { data[i] += other.data[i]; }
	return *this;
}

ml::Tensor& ml::Tensor::operator-=(const ml::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	for(int i = 0; i < dataSize; ++i) { data[i] -= other.data[i]; }
	return *this;
}

ml::Tensor& ml::Tensor::operator*=(const ml::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	for(int i = 0; i < dataSize; ++i) { data[i] *= other.data[i]; }
	return *this;
}

ml::Tensor& ml::Tensor::operator/=(const ml::Tensor& other) {
	if (this->operator!=(other)) {
		throw std::invalid_argument("Invalid given tensor shape");
	}
	for(int i = 0; i < dataSize; ++i) { data[i] /= other.data[i]; }
	return *this;
}

ml::Tensor ml::Tensor::operator+(double value) const {
	ml::Tensor result(*this);
	std::transform(result.data.begin(), result.data.end(), result.data.begin(),
        [value](int x) { return x + value; });
	return result;
}

ml::Tensor ml::Tensor::operator-(double value) const {
	ml::Tensor result(*this);
	std::transform(result.data.begin(), result.data.end(), result.data.begin(),
        [value](int x) { return x - value; });
	return result;
}

ml::Tensor ml::Tensor::operator*(double value) const {
	ml::Tensor result(*this);
	std::transform(result.data.begin(), result.data.end(), result.data.begin(),
        [value](int x) { return x * value; });
	return result;
}

ml::Tensor ml::Tensor::operator/(double value) const {
	ml::Tensor result(*this);
	std::transform(result.data.begin(), result.data.end(), result.data.begin(),
        [value](int x) { return x / value; });
	return result;
}

ml::Tensor& ml::Tensor::operator+=(double value) {
	std::transform(data.begin(), data.end(), data.begin(),
        [value](int x) { return x + value; });
	return *this;
}

ml::Tensor& ml::Tensor::operator-=(double value) {
	std::transform(data.begin(), data.end(), data.begin(),
        [value](int x) { return x - value; });
	return *this;
}

ml::Tensor& ml::Tensor::operator*=(double value) {
	std::transform(data.begin(), data.end(), data.begin(),
        [value](int x) { return x * value; });
	return *this;
}

ml::Tensor& ml::Tensor::operator/=(double value) {
	std::transform(data.begin(), data.end(), data.begin(),
        [value](int x) { return x / value; });
	return *this;
}

int ml::Tensor::calcIndex(const std::vector<int>& indices) const {
    int result = 0, multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        result += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return result;
}