#pragma once

#include "Synapse/AI/interfaces/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn
{
	class Convolutional : public syn::ILayer
	{
	public:
		Convolutional() = default;
		Convolutional(const std::vector<int> &inputShape, int kernelSize, int depth);

		// Declare tunable interface
		void randomize() final;
		void tune(double alpha) final;

		// Declare savable interface
		void save(std::ofstream &file) const;
		void load(std::ifstream &file);

		// Declare clonable interface
		Convolutional *clone() const final;

		// Declare Layer interface
		syn::Tensor forward(const syn::Tensor &inputs) final;
		syn::Tensor backward(const syn::Tensor &outputGrad) final;
		void step(double rate) final;

	private:
		std::vector<int> inputShape;
		int depth, inputDepth, kernelSize;
		syn::Tensor input, output, biases, kernels, kernelsGrad, biasesGrad;
	};
}