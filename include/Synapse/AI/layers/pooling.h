#pragma once

#include "Synapse/AI/interfaces/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn
{
	class Pooling : public syn::ILayer
	{
	public:
		Pooling() = default;
		Pooling(int poolSize, int strides = -1);

		// Declare tunable interface
		void randomize() final {}
		void tune(double alpha) final {}

		// Declare savable interface
		void save(std::ofstream &file) const;
		void load(std::ifstream &file);

		// Declare clonable interface
		Pooling *clone() const final;

		// Declare Layer interface
		syn::Tensor forward(const syn::Tensor &inputs) final;
		syn::Tensor backward(const syn::Tensor &outputGrad) final;
		void step(double rate) final {}

	private:
		int poolSize, strides;
		syn::Tensor input, output;
	};
}