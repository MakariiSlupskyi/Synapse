#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/AI/data.h"
#include <vector>
#include <string>
#include <map>

namespace syn {
    class Model;

    class Optimizer {
    public:
        Optimizer(syn::Model* model, std::vector<syn::Layer*>* layers) : model(model), layers(layers)
        {}

        virtual void train(const syn::Data& inputs, const syn::Data& labels, int epoches, bool printResult = true) = 0;

    protected:
        syn::Model* model;
        std::vector<syn::Layer*>* layers;
    };
}