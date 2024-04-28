#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn {
	class Flatten : public syn::ILayer
	{
	public:
		Flatten() {}
		Flatten(std::ifstream& file) {}

		syn::Tensor forward(const syn::Tensor& input) override;
		syn::Tensor backward(const syn::Tensor& outputGrad) override;

        void clearGradient() override {}
        void update(double learningRate) override {}
    
		void write(std::ofstream& file) const override;

	private:
		std::vector<int> inputShape;
	};
}