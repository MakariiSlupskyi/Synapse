#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/linear.h"

namespace syn {
    class Dense : public syn::ILayer
    {
    public:
        Dense(int nInput, int nOutput);
		Dense(std::ifstream& file);

        syn::Tensor forward(const syn::Tensor& inputs) override;
        syn::Tensor backward(const syn::Tensor& outputsGrad) override;

        void clearGradient() override;
        void update(double learningRate) override;

        void write(std::ofstream& file) const override;

    private:
        syn::Tensor inputs, outputs, biases, weights, weightsGrad, outputsGrad;
    };
}