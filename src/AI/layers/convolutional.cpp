#include "Synapse/AI/layers/Convolutional.h"
#include "Synapse/AI/functions/other_funcs.h"
#include "Synapse/linear.h"
#include <stdexcept>

syn::Convolutional::Convolutional(const std::vector<int>& inputShape, int kernelSize, int depth)
	: inputShape(inputShape), kernelSize(kernelSize), depth(depth)
{
	// Define temperary data
	int inputHeight = inputShape[1];
	int inputWidth = inputShape[2];
	std::vector<int> outputShape = {depth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1};

	// Define fields
	inputDepth = inputShape[0];
	input = syn::Tensor(inputShape);
	output = syn::Tensor(outputShape);
	
	biases = syn::Tensor(outputShape);
	kernels = syn::Tensor({depth, inputDepth, kernelSize, kernelSize});

	biasesGrad = syn::Tensor(outputShape);
	kernelsGrad = syn::Tensor({depth, inputDepth, kernelSize, kernelSize});

	// Randomize layel data
	kernels.randomize();
	biases.randomize();
}

syn::Convolutional::Convolutional(std::ifstream& file) {
	std::string line;

	// Read general layel data
	std::getline(file, line);
	inputShape = syn::splitToI(line);
	
	std::getline(file, line);
	kernelSize = std::stoi(line);

	std::getline(file, line);
	depth = std::stoi(line);

	*this = syn::Convolutional(inputShape, kernelSize, depth);

	// Read biases data
	std::getline(file, line);
	biases.fill(syn::splitToD(line));

	// Read kernels data
	std::getline(file, line);
	kernels.fill(syn::splitToD(line));
}

syn::Tensor syn::Convolutional::forward(const syn::Tensor& input) {
	this->input.fill(input.getData());
	output.fill(biases.getData());

	for (int i = 0; i < depth; ++i) {
		syn::Tensor slice = output.slice({i}), kernel = kernels.slice({i});

		for (int j = 0; j < inputDepth; ++j) {
			slice += syn::correlate2d(input.slice({j}), kernel.slice({j}), "valid");
		}

		output.setSlice({i}, slice);
	}

	return output;
}

syn::Tensor syn::Convolutional::backward(const syn::Tensor& outGrad) {
	syn::Tensor inputGrad(input.getShape());

	for (int i = 0; i < depth; ++i) {
		for (int j = 0; j < inputDepth; ++j) {
			kernelsGrad.setSlice({i, j}, syn::correlate2d(input.slice({j}), outGrad.slice({i}), "valid"));
			inputGrad.setSlice({j}, inputGrad.slice({j}) + syn::convolve2d(outGrad.slice({i}), kernels.slice({i, j}), "full"));
		}
	}
    
	this->kernelsGrad += kernelsGrad;
	this->biasesGrad += outGrad;

	return inputGrad;
}

void syn::Convolutional::randomize() {
	biases.randomize();
	kernels.randomize();
}

void syn::Convolutional::tune(double alpha) {
	biases.tune(alpha);
	kernels.tune(alpha);
}

void syn::Convolutional::clearGradient() {
    kernelsGrad.zeros();
    biasesGrad.zeros();
}

void syn::Convolutional::update(double rate) {
	kernels -= kernelsGrad * rate;
	biases -= biasesGrad * rate;
}
void syn::Convolutional::write(std::ofstream& file) const {
	file << "Convolutional\n";

	for (auto elem : inputShape) { file << elem << ' '; }
	file << '\n';

	file << kernelSize << '\n';

	file << depth << '\n';

	for (auto elem : biases.getData()) { file << elem << ' '; }
	file << '\n';

	for (auto elem : kernels.getData()) { file << elem << ' '; }
	file << '\n';
}