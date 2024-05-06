#pragma once

#include "Synapse/AI/interfaces/layer.h"
#include "Synapse/linear.h"

namespace syn
{
    class Dense : public syn::ILayer
    {
    public:
        Dense() = default;
        Dense(int nInput, int nOutput);

        // Declare tunable interface
        void randomize() final;
        void tune(double alpha) final;

        // Declare savable interface
        void save(std::ofstream &file) const;
        void load(std::ifstream &file);

        // Declare clonable interface
        Dense *clone() const final;

        // Declare Layer interface
        syn::Tensor forward(const syn::Tensor &inputs) final;
        syn::Tensor backward(const syn::Tensor &outputGrad) final;
        void step(double rate) final;

    private:
        syn::Tensor input, output, biases, weights, weightsGrad, outputGrad;
    };
}