#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn {
	class Pooling : public syn::Layer
	{
	public:
		Pooling(int poolSize, int strides = -1);
		Pooling(std::ifstream& file);
		
		syn::Tensor forward(const syn::Tensor& inputs) override;
        syn::Tensor backward(const syn::Tensor& outGrad) override;

        void clearGradient() override {}
        void update(double learningRate) override {}
    
        void write(std::ofstream& file) const override;

	private:
		int poolSize, strides;
		syn::Tensor inputs, outputs;
	};
}