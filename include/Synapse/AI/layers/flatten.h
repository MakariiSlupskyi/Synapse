#pragame once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear/tensor.h"

namespace syn {
	class Flatten : public syn::Layer
	{
	public:
		Flatten();
		Flatten(std::ifstream& file);

		syn::Tensor forward(const syn::Tensor& input) override;
		syn::Tensor backward(const syn::Tensor& outputGrad, double learningRate) override;

    	syn::Tensor getParameters() override { return syn::Tensor({0}); };
		void setParameters(const syn::Tensor& other) {};

		void write(std::ofstream& file) const override;

	private:
		std::vector<int> inputShape;
	};
}