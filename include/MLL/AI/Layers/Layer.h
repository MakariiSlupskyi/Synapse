#pragma once

#include "MLL/Linear/Tensor.h"

namespace ml {
    class Layer
    {
    public:
        virtual ml::Tensor forward(const ml::Tensor& inputs) = 0;
        virtual ml::Tensor backward(const ml::Tensor& outGrad) = 0;
        virtual void update(double learningRate) = 0;
    };
}