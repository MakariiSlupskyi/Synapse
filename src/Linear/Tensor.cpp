#include "Tensor.h"

#include <omp.h>
#include <execution>

// FIXME
// #include <iostream>

ml::Tensor::Tensor() : shape({1}), data({1})
{}

ml::Tensor::Tensor(const std::vector<size_t>& shape)
    : shape(shape), dataSize(1)
{
    for (int i = 0; i < shape.size(); ++i) { dataSize *= shape[i]; }
    data.resize(dataSize);
}

ml::Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& data)
    : Tensor(shape)
{
    if (data.size() != dataSize) {
        throw std::runtime_error("Invalid tensor data size!");
    }
    this->data = data;
}

double ml::Tensor::operator()(const std::vector<size_t>& indices) const {
    return data[0];
}

double& ml::Tensor::operator()(const std::vector<size_t>& indices) {
    return data[0];
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

ml::Tensor& ml::Tensor::randomize() {
	for (int i = 0; i < dataSize; ++i) { data[i] = 0.0d; }
    return *this;
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
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] += other.data[i]; }
	return res;
}

ml::Tensor ml::Tensor::operator-(const ml::Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] -= other.data[i]; }
	return res;
}

ml::Tensor ml::Tensor::operator*(const ml::Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] *= other.data[i]; }
	return res;
}

ml::Tensor ml::Tensor::operator/(const ml::Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data[i] /= other.data[i]; }
	return res;
}

ml::Tensor& ml::Tensor::operator+=(const ml::Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	for(int i = 0; i < dataSize; ++i) { data[i] += other.data[i]; }
	return *this;
}

ml::Tensor& ml::Tensor::operator-=(const ml::Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	for(int i = 0; i < dataSize; ++i) { data[i] -= other.data[i]; }
	return *this;
}

ml::Tensor& ml::Tensor::operator*=(const ml::Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	for(int i = 0; i < dataSize; ++i) { data[i] *= other.data[i]; }
	return *this;
}

ml::Tensor& ml::Tensor::operator/=(const ml::Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid given tensor shape"); }
	for(int i = 0; i < dataSize; ++i) { data[i] /= other.data[i]; }
	return *this;
}

int ml::Tensor::calcIndex(const std::vector<size_t>& indices) const {
    int result = 0, multiplier = 1;
    for (int i = 0; i < shape.size(); ++i) {
        result += indices.at(i) * multiplier;
        multiplier *= shape.at(i);
    }
    return result;
}