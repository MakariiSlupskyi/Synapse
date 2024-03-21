#pragame once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn {
	class Pooling : public syn::Layer
	{
	public:
		Pooling(int poolSize, int strides = -1);
		Pooling(std::ifstream& file);
	
		syn::Tensor forward(const syn::Tensor& input) override;
		syn::Tensor backward(const syn::Tensor& outputGrad, double learningRate) override;
		
        syn::Tensor getParameters() override { return syn::Tensor({0}); };
		void setParameters(const syn::Tensor& other) {};

		void write(std::ofstream& file) const override;

	private:
		int poolSize, strides;
		syn::Tensor input, output;
	};
}