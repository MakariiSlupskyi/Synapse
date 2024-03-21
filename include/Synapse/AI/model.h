#pragma once

#include "Synapse/AI/layers/layer.h"
#include "Synapse/AI/data.h"
#include "Synapse/AI/optimizers/optimizer.h"
#include <string>

namespace syn {
    class Model
    {
    public:
        Model(const std::vector<syn::Layer*>& layers = {});
        ~Model();
        
        std::string getLossType() const { return lossType; }

        void compile(const std::string& optimType, const std::string& lossFuncType);
    
		double evaluate(const syn::Data& inputs, const syn::Data& labels);
		syn::Tensor predict(const syn::Tensor& inputs);
		void train(const syn::Data& trainingData, const syn::Data& labels, int epoches = 1);

		void save(const std::string& path) const;
		syn::Model load(const std::string& path);

        void backward(const syn::Tensor& loss);
        void update(double rate);

        void print();

    private:
        std::vector<syn::Layer*> layers;
        std::string lossType, optimType;
        syn::Optimizer* optimizer;
    };
}