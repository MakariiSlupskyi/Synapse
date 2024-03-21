#pragame once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn {
    class Convolutional : public Layer
	{
	public:
		Convolutional(const std::vector<int>& inputShape, const std::vector<int>& kernelShape, int depth);
		Convolutional(std::ifstream& file);

		syn::Tensor forward(const syn::Tensor& input) override;
		syn::Tensor backward(const syn::Tensor& outputGrad, double learningRate) override;
		
        syn::Tensor getParameters() override { return kernels; };
		void setParameters(const syn::Tensor& other) { kernels = other; };

		void write(std::ofstream& file) const override;

	private:
		std::vector<int> inputShape, kernelShape;
		int depth, inputDepth;
		syn::Tensor input, output, biases, kernels;
	};
}