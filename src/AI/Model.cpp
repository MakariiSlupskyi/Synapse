#include "MLL/AI/Model.h"

ml::Model::Model(const std::vector<ml::Layer*>& layers) : layers(layers)
{}

double ml::Model::evaluate(const ml::Data& inputs, const ml::Data& labels) {
    return 0.0;
}

ml::Tensor ml::Model::inference(const ml::Tensor& inputs) {
   	ml::Tensor output = inputs;
	for (int i = 0; i < layers.size(); ++i) {
		output = layers[i]->forward(output);
	}
	return output; 
}

void train(const ml::Data& trainingData, const ml::Data& labels, int epoches) {

}