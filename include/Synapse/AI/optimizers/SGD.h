#pragma once

#include "Synapse/AI/optimizers/optimizer.h"
#include "Synapse/AI/functions/loss_funcs.h"

namespace syn {
    class SGD : public syn::Optimizer {
    public:
        SGD(syn::Model* model, std::vector<syn::ILayer*>* layers, double rate = 0.05)
        : syn::Optimizer(model, layers), rate(rate)
        {}

        void train(const syn::Data& inputs, const syn::Data& labels, int epoches, bool printResult = true) override {
            double error = 100000000;
            for (int i = 0; i < epoches; ++i) {
                for (int j = 0; j < inputs.size(); ++j) {
                    // calculating gradient
                    syn::Tensor loss = (labels[j] - model->predict(inputs[j])) * -2.0;
                    model->backward(loss / inputs.size());
                    
                    // updating parameters
                    model->update(rate);
                    
                    double temp = model->evaluate(inputs, labels);
                    if (error > temp) { error = temp; }
                    else if (error - temp > 1.0) { return; }

                    error = model->evaluate(inputs, labels);
                }
                // print results
                if (printResult) {
                    std::cout << "epoch: " << i + 1 << " loss: " << error << '\n';
                }
            }
        }
    
    private:
        double rate;
    };
}