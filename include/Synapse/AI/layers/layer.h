#pragma once

#include "Synapse/linear/tensor.h"
#include <fstream>

namespace syn {
    class ILayer
    {
    public:
        virtual syn::Tensor forward(const syn::Tensor& inputs) = 0;
        virtual syn::Tensor backward(const syn::Tensor& outGrad) = 0;

        virtual void clearGradient() = 0;
        virtual void update(double learningRate) = 0;
    
        virtual void write(std::ofstream& file) const = 0;
    };
}