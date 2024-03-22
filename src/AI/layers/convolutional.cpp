#include "Synapse/AI/layers/Convolutional.h"
#include "Synapse/AI/functions/other_funcs.h"

syn::Convolutional::Convolutional(const std::vector<int>& inputShape, const std::vector<int>& kernelShape, int depth)
	: inputShape(inputShape), kernelShape(kernelShape), depth(depth), inputDepth(inputShape[0])
{
	std::vector<int> outputShape = {
		depth,
		inputShape[1] - kernelShape[0] + 1,
		inputShape[2] - kernelShape[1] + 1,
	};
	input = input.reshape(inputShape);

	output = output.reshape(outputShape);
	biases = biases.reshape(outputShape);
	kernels = kernels.reshape({
		depth,
		inputDepth,
		kernelShape[0],
		kernelShape[1],
	});

	kernels.randomize();
	biases.randomize();
}

syn::Convolutional::Convolutional(std::ifstream& file) {
	std::string line;

	for (int i = 0; i <= 2; ++i) {
		std::getline(file, line);

		if (i == 0) {
			inputShape = syn::splitToI(line);
			std::getline(file, line);
			kernelShape = syn::splitToI(line);
			std::getline(file, line);
			depth = std::stoi(line);

			*this = syn::Convolutional(inputShape, kernelShape, depth);
		} else if (i == 1) {
			biases.fill(syn::splitToD(line));
		} else if (i == 2) {
			kernels.fill(syn::splitToD(line));
		}
	}
}

syn::Tensor syn::Convolutional::forward(const syn::Tensor& inputs) {
	output.fill(biases.getData());

	// for (int i = 0; i < depth; ++i) {
	// 	syn::Tensor slice = output.slice({i}), kernel = kernels.slice({i});
	// 	for (int j = 0; j < inputDepth; ++j) {
	// 		slice += syn::correlate2d(input.slice({j}), kernel.slice({j}), "valid");
	// 	}

	// 	output.setSlice({i}, slice);
	// }

	return output;
}

syn::Tensor syn::Convolutional::backward(const syn::Tensor& outGrad) {
	syn::Tensor kernelsGrad(kernels.getShape());
	syn::Tensor inputGrad(input.getShape());

	for (int i = 0; i < depth; ++i) {
		for (int j = 0; j < inputDepth; ++j) {
			// kernelsGrad.setSlice({i, j}, syn::correlate2d(input.slice({j}), outputGrad.slice({i}), "valid"));
			// inputGrad.setSlice({j}, inputGrad.slice({j}) + syn::convolve2d(outputGrad.slice({i}), kernels.slice({i, j}), "full"));
		}
	}
    
	this->kernelsGrad += kernelsGrad;
	this->outputsGrad += outputsGrad;

	return inputGrad;
}


void syn::Convolutional::clearGradient() {
    kernelsGrad.zeros();
    outputsGrad.zeros();
}

void syn::Convolutional::update(double rate) {
	kernels -= kernelsGrad * rate;
	biases -= outputsGrad * rate;
}
void syn::Convolutional::write(std::ofstream& file) const {
	file << "Convolutional" << std::endl;

	for (auto elem : inputShape) { file << elem << ' '; }
	file << std::endl;

	for (auto elem : kernelShape) { file << elem << ' '; }
	file << std::endl;

	file << depth << std::endl;

	for (auto elem : biases.getData()) { file << elem << ' '; }
	file << std::endl;

	for (auto elem : kernels.getData()) { file << elem << ' '; }
	file << std::endl;
}