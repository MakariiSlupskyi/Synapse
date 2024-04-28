#include "Synapse/AI/layers/pooling.h"
#include "Synapse/AI/functions/other_funcs.h"
#include <vector>
#include <string>

syn::Pooling::Pooling(int poolSize, int strides) : poolSize(poolSize), strides(strides == -1 ? poolSize : strides)
{}

syn::Pooling::Pooling(std::ifstream& file) {
	std::string line;
	std::getline(file, line);
	std::vector<int> data = syn::splitToI(line);
	*this = syn::Pooling(data[0], data[1]);
}

syn::Tensor syn::Pooling::forward(const syn::Tensor& inputs) {
    this->inputs = inputs;
	
	outputs = outputs.reshape({inputs.getShape()[0], inputs.getShape()[1] / strides, inputs.getShape()[2] / strides});

	for (int i = 0; i < outputs.getShape()[0]; ++i) {
		for (int j = 0; j < outputs.getShape()[1]; ++j) {
			for (int k = 0; k < outputs.getShape()[2]; ++k) {
				// outputs({i, j, k}) = input.block(
				// 	{i, j * strides, k * strides},
				// 	{1, poolSize, poolSize}
				// ).max();
			}
		}
	}
	return outputs;
}


syn::Tensor syn::Pooling::backward(const syn::Tensor& outputGrad) {
	syn::Tensor inputsGrad(inputs.getShape());
	for (int i = 0; i < outputs.getShape()[0]; ++i) {
		for (int j = 0; j < outputs.getShape()[1]; ++j) {
			for (int k = 0; k < outputs.getShape()[2]; ++k) {
				inputsGrad({i, j * strides, k * strides}) = outputGrad({i, j, k});
			}
		}
	}
	return inputsGrad;
}

void syn::Pooling::write(std::ofstream& file) const {
	file << "Pooling" << std::endl;
	file << poolSize << ' ' << strides << std::endl;
}