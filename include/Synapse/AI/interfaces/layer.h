#pragma once

#include "Synapse/AI/interfaces/clonable.h"
#include "Synapse/AI/interfaces/savable.h"
#include "Synapse/AI/interfaces/tunable.h"
#include "Synapse/linear/tensor.h"

namespace syn
{
    class ILayer : public syn::ISavable, public syn::ITunable, public syn::IClonable
    {
    public:
        virtual syn::Tensor forward(const syn::Tensor &inputs) = 0;
        virtual syn::Tensor backward(const syn::Tensor &outputGrad) = 0;
        virtual void step(double rate) = 0;
    };
}