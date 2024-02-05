#pragma once

#include "MLL/AI/Layers/Layer.h"
#include "MLL/AI/Data.h"

namespace ml {
    class Model
    {
    public:
        Model(const std::vector<ml::Layer*>& layers = {});
    
		double evaluate(const ml::Data& inputs, const ml::Data& labels);
		ml::Tensor inference(const ml::Tensor& inputs);
		void train(const ml::Data& trainingData, const ml::Data& labels, int epoches);

    private:
        std::vector<ml::Layer*> layers;
    };
}