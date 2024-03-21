#pragma once

#include "Synapse/AI/optimizers/optimizer.h"
#include "Synapse/AI/functions/loss_funcs.h"
#include "Synapse/AI/model.h"
#include <iostream>

namespace syn {
    class GD : public syn::Optimizer {
    public:
        GD(syn::Model* model, std::vector<syn::Layer*>* layers, double rate = 0.1)
        : syn::Optimizer(model, layers), rate(rate)
        {}

        void train(const syn::Data& inputs, const syn::Data& labels, int epoches, bool printResult = true) override {
            for (int i = 0; i < epoches; ++i) {
                syn::Tensor loss = labels[0].zeros();

                // calculating gradient
                for (int j = 0; j < inputs.size(); ++j) {
                    syn::Tensor loss = (labels[j] - model->predict(inputs[j])) * -2.0;
                    model->backward(loss);
                }

                // updating parameters
                model->update(rate);
            
                // print results
                if (printResult) {
                    std::cout << "epoch: " << i + 1 << " loss: " << model->evaluate(inputs, labels) << '\n';
                }
            }
        }
    
    private:
        double rate;
    };
}