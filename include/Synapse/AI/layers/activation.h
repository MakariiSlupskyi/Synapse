#pragma once

#include "Synapse/AI/layers/layer.h"
#include <string>

namespace syn {
    class Activation : public syn::ILayer
    {
    public:
        Activation(const std::string& type);
		Activation(std::ifstream& file);

		syn::Tensor forward(const syn::Tensor& inputs) override;
		syn::Tensor backward(const syn::Tensor& outGrad) override;

        void clearGradient() override {}
        void update(double learningRate) override {}

		void write(std::ofstream& file) const override;

    private:
        std::string type;
        syn::Tensor (*func)(const syn::Tensor& tensor);
        syn::Tensor (*prime)(const syn::Tensor& tensor);
        syn::Tensor inputs;
    };
}