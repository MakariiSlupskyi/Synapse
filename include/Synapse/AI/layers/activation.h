#pragma once

#include "Synapse/AI/interfaces/layer.h"
#include <string>

namespace syn
{
    class Activation : public syn::ILayer
    {
    public:
        Activation() = default;
        Activation(const std::string &type);

        // Declare tunable interface
        void randomize() final {}
        void tune(double alpha) final {}

        // Declare savable interface
        void save(std::ofstream &file) const;
        void load(std::ifstream &file);

        // Declare clonable interface
        Activation *clone() const final;

        // Declare Layer interface
        syn::Tensor forward(const syn::Tensor &inputs) final;
        syn::Tensor backward(const syn::Tensor &outputGrad) final;
        void step(double rate) final {}

    private:
        std::string type;
        syn::Tensor (*activFunc)(const syn::Tensor &tensor);
        syn::Tensor (*activPrime)(const syn::Tensor &tensor);
        syn::Tensor input;
    };
}