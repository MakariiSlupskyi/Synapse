#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn {
    class Convolutional : public syn::ILayer
	{
	public:
		Convolutional(const std::vector<int>& inputShape, int kernelSize, int depth);
		Convolutional(std::ifstream& file);

        void randomize();
        void tune(double alpha);

        syn::Tensor forward(const syn::Tensor& inputs) override;
        syn::Tensor backward(const syn::Tensor& outGrad) override;

        void clearGradient() override;
        void update(double learningRate) override;
    
        void write(std::ofstream& file) const override;

	private:
		std::vector<int> inputShape;
		int depth, inputDepth, kernelSize;
		syn::Tensor biases, kernels, input, output, kernelsGrad, biasesGrad;
	};
}