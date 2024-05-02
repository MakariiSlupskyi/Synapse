#include "Synapse/linear/tensor.h"

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <functional>
#include <execution>

syn::Tensor::Tensor() : shape({1}), data({1}), dataSize(1)
{
}

syn::Tensor::Tensor(const std::vector<int> &shape)
	: shape(shape), dataSize(1)
{
	for (int i = 0; i < shape.size(); ++i)
	{
		dataSize *= shape[i];
	}
	data.resize(dataSize);
}

syn::Tensor::Tensor(const std::vector<int> &shape, const std::vector<double> &data)
	: Tensor(shape)
{
	if (data.size() != dataSize)
	{
		throw std::runtime_error("Invalid size of given tensor data");
	}
	this->data = data;
}

syn::Tensor &syn::Tensor::fill(double value)
{
	std::fill(data.begin(), data.end(), value);
	return *this;
}

syn::Tensor &syn::Tensor::fill(const std::vector<double> &values)
{
	if (data.size() != dataSize)
	{
		throw std::runtime_error("Invalid size of given tensor data");
	}
	for (int i = 0; i < dataSize; ++i)
	{
		data[i] = values[i];
	}
	return *this;
}

syn::Tensor &syn::Tensor::zeros()
{
	std::fill(data.begin(), data.end(), 0.0);
	return *this;
}

syn::Tensor &syn::Tensor::ones()
{
	std::fill(data.begin(), data.end(), 1.0);
	return *this;
}

#include <iostream>
syn::Tensor &syn::Tensor::randomize()
{
	for (int i = 0; i < data.size(); ++i)
	{
		data[i] = getRandom(-1.0, 1.0);
		std::cout << data[i] << ' ';
	}
	std::cout << std::endl;
	return *this;
}

syn::Tensor &syn::Tensor::tune(double alpha)
{
	for (int i = 0; i < data.size(); ++i)
	{
		data[i] += getRandom(-alpha, alpha);
		std::cout << data[i] << ' ';
	}
	std::cout << std::endl;
	return *this;
}

syn::Tensor &syn::Tensor::reshape(const std::vector<int> &shape)
{
	// Calculate new size of data
	int dataSize = 1;
	for (int i = 0; i < shape.size(); ++i)
	{
		dataSize *= shape[i];
	}

	// If new size of data doesn't match with previous one, zero data
	if (this->dataSize != dataSize)
	{
		this->data = std::vector<double>(dataSize);
	}

	this->dataSize = dataSize;
	this->shape = shape;

	return *this;
}

syn::Tensor &syn::Tensor::reverse()
{
	std::reverse(this->data.begin(), this->data.end());
	return *this;
}

double syn::Tensor::sum() const
{
	return std::accumulate(data.begin(), data.end(), 0.0);
}

double syn::Tensor::max() const
{
	return *std::max_element(data.begin(), data.end());
}

double syn::Tensor::min() const
{
	return *std::min_element(data.begin(), data.end());
}

double syn::Tensor::at(const std::vector<int> &indices) const
{
	return data[getDataIndex(indices)];
}

double &syn::Tensor::at(const std::vector<int> &indices)
{
	return data[getDataIndex(indices)];
}

syn::Tensor syn::Tensor::chip(int axisIndex, int index) const
{
	if (axisIndex > shape.size() || index > shape[axisIndex])
	{
		throw std::invalid_argument("Invalid given indices for chipping tensor.");
	}

	// Prepare result tensor
	std::vector<int> resShape(shape);
	resShape.erase(resShape.begin() + axisIndex);
	syn::Tensor result(resShape);

	std::vector<int> resInds(shape.size() - 1), thisInds(shape.size());
	thisInds[axisIndex] = index;
	for (int i = 0; i < result.dataSize; ++i)
	{
		for (int j = 0; j < resInds.size(); ++j)
		{
			thisInds.at((j < axisIndex) ? j : j + 1) = resInds.at(j);
		}

		result(resInds) = this->operator()(thisInds);
		result.increaseIndices(resInds);
	}
	return result;
}

syn::Tensor syn::Tensor::slice(const std::vector<int> &indices) const
{
	if (indices.size() >= shape.size())
	{
		throw std::invalid_argument("Invalid given indices for slicing tensor.");
	}

	if (indices.size() == 0)
	{
		return *this;
	}

	// Get a chip
	syn::Tensor res = this->chip(0, indices.at(0));

	if (indices.size() == 1)
	{
		return res;
	}
	else
	{
		std::vector<int> nextIndices(indices.cbegin() + 1, indices.cend());
		return res.slice(nextIndices);
	}
}

syn::Tensor syn::Tensor::block(const std::vector<int> &start, const std::vector<int> &blockShape) const
{
	if (start.size() != shape.size() || blockShape.size() != shape.size())
	{
		throw std::invalid_argument("Invalid arguments for getting block of a tensor.");
	}

	syn::Tensor res(blockShape);
	std::vector<int> resInds(shape.size()), thisInds(shape.size());
	for (int i = 0; i < res.dataSize; ++i)
	{
		for (int j = 0; j < blockShape.size(); ++j)
		{
			thisInds[j] = resInds[j] + start[j];
		}
		res(resInds) = this->operator()(thisInds);
		res.increaseIndices(resInds);
	}
	return res;
}

syn::Tensor &syn::Tensor::setChip(int axisInd, int index, const Tensor &other)
{
	std::vector<int> thisInds(shape.size(), 0), otherInds(shape.size() - 1, 0);
	thisInds[axisInd] = index;
	for (int i = 0; i < other.dataSize; ++i)
	{
		for (int j = 0; j < otherInds.size(); ++j)
		{
			thisInds.at((j < axisInd) ? j : j + 1) = otherInds.at(j);
		}
		this->operator()(thisInds) = other(otherInds);
		other.increaseIndices(otherInds);
	}
	return *this;
}

syn::Tensor &syn::Tensor::setSlice(const std::vector<int> &indices, const Tensor &other)
{
	if (indices.size() == 0)
	{
		this->data = other.data;
	}
	else
	{
		std::vector<int> indices_(indices.cbegin() + 1, indices.cend());
		this->setChip(0, indices[0], this->chip(0, indices[0]).setSlice(indices_, other));
	}
	return *this;
}

syn::Tensor &syn::Tensor::setBlock(const std::vector<int> &start, const Tensor &other)
{
	std::vector<int> thisInds(shape.size(), 0), blockInds(shape.size(), 0);
	for (int i = 0; i < other.dataSize; ++i)
	{
		for (int j = 0; j < thisInds.size(); ++j)
		{
			thisInds.at(j) = start.at(j) + blockInds.at(j);
		}
		this->operator()(thisInds) = other(blockInds);
		other.increaseIndices(blockInds);
	}
	return *this;
}

syn::Tensor syn::Tensor::matMul(const syn::Tensor &other) const
{
	if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0])
	{
		throw std::runtime_error("Invalid given tensor shape for matrix multiplication");
	}
	syn::Tensor result({shape[0], other.shape[1]});
	result.fill(0.0);
	for (int i = 0; i < shape[0]; ++i)
	{
		for (int k = 0; k < shape[1]; ++k)
		{
			for (int j = 0; j < other.shape[1]; ++j)
			{
				result({i, j}) += (*this)({i, k}) * other({k, j});
			}
		}
	}
	return result;
}

syn::Tensor syn::Tensor::matTrans() const
{
	if (shape.size() != 2)
	{
		throw std::runtime_error("Invalid given tensor shape for matrix transposing");
	}
	const int nRow = shape[0];
	const int nCol = shape[1];
	syn::Tensor result({nCol, nRow});
	for (int i = 0; i < nRow; ++i)
	{
		for (int j = 0; j < nCol; ++j)
		{
			result({j, i}) = this->operator()({i, j});
		}
	}
	return result;
}

syn::Tensor &syn::Tensor::apply(double (*func)(double))
{
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(), func);
	return *this;
}

syn::Tensor syn::Tensor::apply(double (*func)(double)) const
{
	syn::Tensor res(*this);
	std::transform(std::execution::par, res.data.begin(), res.data.end(), res.data.begin(), func);
	return res;
}

syn::Tensor &syn::Tensor::square()
{
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
				   [](double x) -> double
				   { return x * x; });
	return *this;
}

syn::Tensor &syn::Tensor::sqrt()
{
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
				   [](double x) -> double
				   { return x * x; });
	return *this;
}

syn::Tensor &syn::Tensor::abs()
{
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
				   [](double x) -> double
				   { return std::abs(x); });
	return *this;
}

syn::Tensor &syn::Tensor::log()
{
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
				   [](double x) -> double
				   { return std::log(x); });
	return *this;
}

syn::Tensor &syn::Tensor::exp()
{
	std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
				   [](double x) -> double
				   { return std::exp(x); });
	return *this;
}

double syn::Tensor::operator()(const std::vector<int> &indices) const
{
	for (int i = 0; i < indices.size(); ++i)
	{
		if (shape[i] - 1 < indices[i])
		{
			throw std::runtime_error("Invalid given indices for tensor indexing");
		}
	}
	return data[getDataIndex(indices)];
}

double &syn::Tensor::operator()(const std::vector<int> &indices)
{
	for (int i = 0; i < indices.size(); ++i)
	{
		if (shape[i] - 1 < indices[i])
		{
			throw std::runtime_error("Invalid given indices for tensor indexing");
		}
	}
	return data[getDataIndex(indices)];
}

bool syn::Tensor::operator==(const syn::Tensor &other) const
{
	if (shape.size() != other.shape.size())
	{
		return false;
	}
	for (int i = 0; i < shape.size(); ++i)
	{
		if (shape[i] != other.shape[i])
		{
			return false;
		}
	}
	return true;
}

bool syn::Tensor::operator!=(const syn::Tensor &other) const
{
	return !this->operator==(other);
}

syn::Tensor syn::Tensor::operator+(const syn::Tensor &other) const
{
	if (this->operator!=(other))
	{
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

syn::Tensor syn::Tensor::operator-(const syn::Tensor &other) const
{
	if (this->operator!=(other))
	{
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

syn::Tensor syn::Tensor::operator*(const syn::Tensor &other) const
{
	if (this->operator!=(other))
	{
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

syn::Tensor syn::Tensor::operator/(const syn::Tensor &other) const
{
	if (this->operator!=(other))
	{
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

syn::Tensor &syn::Tensor::operator+=(const syn::Tensor &other)
{
	if (this->operator!=(other))
	{
		throw std::runtime_error("Invalid given tensor shape for addition");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::plus<double>());
	return *this;
}

syn::Tensor &syn::Tensor::operator-=(const syn::Tensor &other)
{
	if (this->operator!=(other))
	{
		throw std::runtime_error("Invalid given tensor shape for subtracting");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::minus<double>());
	return *this;
}

syn::Tensor &syn::Tensor::operator*=(const syn::Tensor &other)
{
	if (this->operator!=(other))
	{
		throw std::runtime_error("Invalid given tensor shape for multiplying");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::multiplies<double>());
	return *this;
}

syn::Tensor &syn::Tensor::operator/=(const syn::Tensor &other)
{
	if (this->operator!=(other))
	{
		throw std::runtime_error("Invalid given tensor shape for division");
	}
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(),
		other.data.begin(), data.begin(),
		std::divides<double>());
	return *this;
}

syn::Tensor syn::Tensor::operator+(double value) const
{
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
		[value](double x)
		{ return x + value; });
	return result;
}

syn::Tensor syn::Tensor::operator-(double value) const
{
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
		[value](double x)
		{ return x - value; });
	return result;
}

syn::Tensor syn::Tensor::operator*(double value) const
{
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
		[value](double x)
		{ return x * value; });
	return result;
}

syn::Tensor syn::Tensor::operator/(double value) const
{
	syn::Tensor result(*this);
	std::transform(
		std::execution::par_unseq,
		result.data.begin(), result.data.end(), result.data.begin(),
		[value](double x)
		{ return x / value; });
	return result;
}

syn::Tensor &syn::Tensor::operator+=(double value)
{
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
		[value](double x)
		{ return x + value; });
	return *this;
}

syn::Tensor &syn::Tensor::operator-=(double value)
{
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
		[value](double x)
		{ return x - value; });
	return *this;
}

syn::Tensor &syn::Tensor::operator*=(double value)
{
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
		[value](double x)
		{ return x * value; });
	return *this;
}

syn::Tensor &syn::Tensor::operator/=(double value)
{
	std::transform(
		std::execution::par_unseq,
		data.begin(), data.end(), data.begin(),
		[value](double x)
		{ return x / value; });
	return *this;
}

int syn::Tensor::getDataIndex(const std::vector<int> &indices) const
{
	int result = 0, multiplier = 1;
	for (int i = shape.size() - 1; i >= 0; --i)
	{
		result += indices[i] * multiplier;
		multiplier *= shape[i];
	}
	return result;
}

double syn::Tensor::getRandom(double min, double max)
{
	float res = float(rand()) / float(RAND_MAX);
	return min + res * (max - min);
}

void syn::Tensor::increaseIndices(std::vector<int> &indices) const
{
	indices.back() += 1;
	for (int i = (int)shape.size() - 1; i >= 0; --i)
	{
		if (indices.at(i) < shape.at(i))
		{
			return;
		}

		indices.at(i) = 0;
		if (i != 0)
		{
			indices.at(i - 1) += 1;
		}
	}
	return;
}

std::ostream &operator<<(std::ostream &os, const syn::Tensor &tensor)
{
	// Define tensor data
	auto shape = tensor.getShape();
	auto data = tensor.getData();
	int dataIndex = 0;

	// Recursive helper function to print nested elements
	std::function<void(int)> printSubtensor = [&](int dim)
	{
		if (dim == shape.size())
		{
			return;
		}
		else if (dim == shape.size() - 1)
		{
			for (int i = 0; i < shape[dim]; ++i)
			{
				os << data[dataIndex];

				if (i < shape[dim] - 1)
				{
					os << ", ";
				}

				++dataIndex;
			}
			return;
		}
		for (int i = 0; i < shape[dim]; ++i)
		{
			os << "[";
			printSubtensor(dim + 1); // Recursive call for subtensors
			os << "]";
			if (i < shape[dim] - 1)
			{
				os << ", ";
			}
		}
	};

	// Actually start printing
	os << "[";
	printSubtensor(0);
	os << "]";

	return os;
}