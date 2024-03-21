#pragma once

#include <vector>

namespace syn {
    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<int>& shape);
        Tensor(const std::vector<int>& shape, const std::vector<double>& data);

        std::vector<int> getShape() const { return shape; }
        std::vector<double> getData() const { return data; }
        
        // Data changing methods
        Tensor& fill(double value);
        Tensor& fill(const std::vector<double>& values);
        Tensor& zeros();
        Tensor& ones();
        Tensor& randomize();
		Tensor& reshape(const std::vector<int>& shape);
		Tensor& reverse();	
	
        // Aggregate functions
		double sum() const;
		double max() const;
		double min() const;

		// Matrix operations
		Tensor matMul(const Tensor& other) const;
		Tensor matTrans() const;
		
        // Element-wise operations and operators
		Tensor& apply(double (*func)(double));
		Tensor apply(double (*func)(double)) const;

		Tensor& square();
		Tensor& sqrt();
		Tensor& abs();
		Tensor& log();
		Tensor& exp();

		// Indexing


        // Operators
        double operator()(const std::vector<int>& indices) const;
        double& operator()(const std::vector<int>& indices);
		
		bool operator==(const Tensor& other) const;
		bool operator!=(const Tensor& other) const;

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