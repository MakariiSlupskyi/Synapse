#include "Synapse/AI/layers/dense.h"
#include "Synapse/AI/functions/other_funcs.h"

syn::Dense::Dense(int nInput, int nOutput)
    : inputs({nInput, 1}), outputs({nOutput, 1}),
    biases({nOutput, 1}), weights({nOutput, nInput}),
    outputsGrad({nOutput, 1}), weightsGrad({nOutput, nInput})
{
    biases.randomize();
    weights.randomize();
    clearGradient();
}

syn::Dense::Dense(std::ifstream& file) {
	std::string line;

	std::getline(file, line);

    std::vector<int>data = syn::splitToI(line);

    *this = syn::Dense(data[0], data[1]);

	std::getline(file, line);
	biases.fill(syn::splitToD(line));

	std::getline(file, line);
	weights.fill(syn::splitToD(line));
}

syn::Tensor syn::Dense::forward(const syn::Tensor& inputs) {
    this->inputs = inputs;
    outputs = weights.matMul(inputs);
    outputs += biases;
    return outputs;
}

syn::Tensor syn::Dense::backward(const syn::Tensor& outputsGrad) {
    weightsGrad = outputsGrad.matMul(inputs.matTrans());

    this->outputsGrad += outputsGrad;
	this->weightsGrad += weightsGrad;

    return weights.matTrans().matMul(outputsGrad);
}

void syn::Dense::clearGradient() {
    weightsGrad.zeros();
    outputsGrad.zeros();
}

void syn::Dense::update(double rate) {
    weights -= weightsGrad * rate;
    biases -= outputsGrad * rate;
}

void syn::Dense::write(std::ofstream& file) const {
	file << "Dense\n";

	file << inputs.getShape()[0] << ' ' << outputs.getShape()[0] << '\n';

	for (auto elem : biases.getData()) { file << elem << ' '; }
	file << '\n';

	for (auto elem : weights.getData()) { file << elem << ' '; }
	file << '\n';
}