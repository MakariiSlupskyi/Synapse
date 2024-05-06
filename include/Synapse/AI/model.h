#pragma once

#include "Synapse/AI/interfaces/layer.h"
#include "Synapse/AI/interfaces/savable.h"
#include "Synapse/AI/data.h"
#include "Synapse/AI/optimizers/optimizer.h"
#include "Synapse/AI/automated/SL/model_builder.h"
#include <memory>
#include <string>

namespace syn
{
    class Model : public syn::ISavable, public syn::ITunable
    {
    public:
        Model(const std::vector<syn::ILayer *> &layers = {});
        Model(const syn::ModelBuilder &builder);
        Model(const syn::Model &other);

        std::string getLossType() const { return lossType; }

        // Set loss function and optimizer
        void compile(const std::string &optimType, const std::string &lossFuncType);

        // Get loss
        double evaluate(const syn::Data &inputs, const syn::Data &labels);

        syn::Tensor predict(const syn::Tensor &inputs);
        void train(const syn::Data &trainingData, const syn::Data &labels, int epoches = 1, bool printLoss = 0);

        void backward(const syn::Tensor &loss);
        void update(double rate);

        void save(const std::string &path) const;
        syn::Model &load(const std::string &path);

        // Declare tunable interface
        void randomize() final;
        void tune(double alpha) final;

        Model clone() const;

    private:
        // Declare savable interface
        void save(std::ofstream &file) const final;
        void load(std::ifstream &file) final;

        std::vector<syn::ILayer *> layers;
        std::string lossType, optimType;
        syn::Optimizer *optimizer;
    };
}