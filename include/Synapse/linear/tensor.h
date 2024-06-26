#pragma once

#include <vector>
#include <ostream>

namespace syn
{
	class Tensor
	{
	public:
		Tensor();
		Tensor(const std::vector<int> &shape);
		Tensor(const std::vector<int> &shape, const std::vector<double> &data);

		std::vector<int> getShape() const { return shape; }
		std::vector<double> getData() const { return data; }

		// Data changing methods
		Tensor &fill(double value);
		Tensor &fill(const std::vector<double> &values);
		Tensor &zeros();
		Tensor &ones();
		Tensor &randomize();
		Tensor &tune(double alpha = 0.1);
		Tensor &reverse();
		Tensor &reshape(const std::vector<int> &shape);

		// indexing
		double at(const std::vector<int> &indices) const;
		double &at(const std::vector<int> &indices);

		Tensor chip(int axisIndex, int index) const;
		Tensor slice(const std::vector<int> &indices) const;
		Tensor block(const std::vector<int> &start, const std::vector<int> &blockShape) const;

		Tensor &setChip(int axisIndex, int index, const Tensor &other);
		Tensor &setSlice(const std::vector<int> &indices, const Tensor &other);
		Tensor &setBlock(const std::vector<int> &start, const Tensor &other);

		// Aggregate functions
		double sum() const;
		double max() const;
		double min() const;

		// Matrix operations
		/// @brief Matrix multiplication
		Tensor matMul(const Tensor &other) const;

		/// @brief Matrix transposing
		Tensor matTrans() const;

		// Element-wise operations
		Tensor &apply(double (*func)(double));
		Tensor apply(double (*func)(double)) const;

		Tensor &square();
		Tensor &sqrt();
		Tensor &abs();
		Tensor &log();
		Tensor &exp();

		// Operators
		double operator()(const std::vector<int> &indices) const;
		double &operator()(const std::vector<int> &indices);

		bool operator==(const Tensor &other) const;
		bool operator!=(const Tensor &other) const;

		Tensor operator+(const Tensor &other) const;
		Tensor operator-(const Tensor &other) const;
		Tensor operator*(const Tensor &other) const;
		Tensor operator/(const Tensor &other) const;

		Tensor &operator+=(const Tensor &other);
		Tensor &operator-=(const Tensor &other);
		Tensor &operator*=(const Tensor &other);
		Tensor &operator/=(const Tensor &other);

		Tensor operator+(double value) const;
		Tensor operator-(double value) const;
		Tensor operator*(double value) const;
		Tensor operator/(double value) const;

		Tensor &operator+=(double value);
		Tensor &operator-=(double value);
		Tensor &operator*=(double value);
		Tensor &operator/=(double value);

	protected:
		std::vector<int> shape;
		std::vector<double> data;
		int dataSize;

	private:
		double getRandom(double min, double max);
		int getDataIndex(const std::vector<int> &indices) const;
		void increaseIndices(std::vector<int> &indices) const;
	};
}

std::ostream &operator<<(std::ostream &os, const syn::Tensor &tensor);