#pragma once

#include <initializer_list>
#include <vector>

namespace ml {
    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<int>& shape);
        Tensor(const std::vector<int>& shape, const std::vector<double>& data);

        std::vector<int> getShape() const { return shape; }
        std::vector<double> getData() const { return data; }
        
        // Data setting methods
        Tensor& fill(double value);
        Tensor& fill(const std::vector<double>& values);
        Tensor& randomize();

        // Indexing tensor methods
        double operator()(const std::vector<int>& indices) const;
        double& operator()(const std::vector<int>& indices);

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

		Tensor operator+(double value) const;
		Tensor operator-(double value) const;
		Tensor operator*(double value) const;
		Tensor operator/(double value) const;

		Tensor& operator+=(double value);
		Tensor& operator-=(double value);
		Tensor& operator*=(double value);
		Tensor& operator/=(double value);

    protected:
        std::vector<int> shape;
        std::vector<double> data;
        int dataSize;

	private:
		int calcIndex(const std::vector<int>& indices) const;
		std::vector<int>& increaseIndices(std::vector<int>& indices) const;
    };
}