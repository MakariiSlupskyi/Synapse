#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/AI/data.h"
#include "Synapse/AI/optimizers/optimizer.h"
#include <cstdlib>
#include <string>


namespace syn {
    class ModelBuilder {
    public:
        ModelBuilder(const syn::Data& inputs, const syn::Data& labels);
        ModelBuilder() {}

        syn::Model build() const;

        syn::ILayer** getLayers() const { return layers; }
        int getNumLayer() const { return nLayer; }
        std::string getLossType() const { return lossType; }
        std::string getOptimType() const { return optimType; }

    protected:
        syn::ILayer** layers;
        int nLayer;
        std::string lossType, optimType;
    };


    class CNNBuilder : public ModelBuilder {
    public:
        CNNBuilder(const syn::Data& inputs, const syn::Data& labels);
    };


    class FFNNBuilder : public ModelBuilder {
    public:
        FFNNBuilder(const syn::Data& inputs, const syn::Data& labels);
    };
}