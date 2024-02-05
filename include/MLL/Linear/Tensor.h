#pragma once

#include <initializer_list>
#include <vector>

namespace ml {
    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<size_t>& shape);
        Tensor(const std::vector<size_t>& shape, const std::vector<double>& data);

        std::vector<size_t> getShape() const { return shape; }
        std::vector<double> getData() const { return data; }
        
        // Set data
        Tensor& fill(double value);
        Tensor& fill(const std::vector<double>& values);
        Tensor& randomize();

        // Get access to tensor element
        double operator()(const std::vector<size_t>& indices) const;
        double& operator()(const std::vector<size_t>& indices);

        // Aggregate functions
		double sum() const;
		double max() const;
		double min() const;

        // Сomparison functions
		bool operator==(const Tensor& other) const;
		bool operator!=(const Tensor& other) const;

        // Element-wise operations and operators
		Tensor& applyFunc(double (*func)(double));
		Tensor applyFunc(double (*func)(double)) const;

        Tensor operator+(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;
		Tensor operator*(const Tensor& other) const;
		Tensor operator/(const Tensor& other) const;

		Tensor& operator+=(const Tensor& other);
		Tensor& operator-=(const Tensor& other);
		Tensor& operator*=(const Tensor& other);
		Tensor& operator/=(const Tensor& other);

		Tensor operator+(double scalar) const;
		Tensor operator-(double scalar) const;
		Tensor operator*(double scalar) const;
		Tensor operator/(double scalar) const;

		Tensor& operator+=(double scalar);
		Tensor& operator-=(double scalar);
		Tensor& operator*=(double scalar);
		Tensor& operator/=(double scalar);

    protected:
        std::vector<size_t> shape;
        std::vector<double> data;
        size_t dataSize;

	private:
		int calcIndex(const std::vector<size_t>& indices) const;
		std::vector<int>& increaseIndices(std::vector<size_t>& indices) const;
    };
}