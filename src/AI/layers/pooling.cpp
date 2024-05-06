#include "Synapse/AI/layers/pooling.h"
#include "Synapse/AI/functions/other_funcs.h"
#include <vector>
#include <string>

syn::Pooling::Pooling(int poolSize, int strides)
	: poolSize(poolSize), strides(strides == -1 ? poolSize : strides)
{
}

void syn::Pooling::save(std::ofstream &file) const
{
	file << "Pooling" << std::endl;
	file << poolSize << ' ' << strides << std::endl;
}

void syn::Pooling::load(std::ifstream &file)
{
	std::string line;
	std::getline(file, line);
	std::vector<int> data = syn::splitToI(line);
	*this = syn::Pooling(data[0], data[1]);
}

syn::Pooling *syn::Pooling::clone() const
{
	return new syn::Pooling(poolSize, strides);
}

syn::Tensor syn::Pooling::forward(const syn::Tensor &input)
{
	this->input = input;

	output = output.reshape({input.getShape()[0], input.getShape()[1] / strides, input.getShape()[2] / strides});

	for (int i = 0; i < output.getShape()[0]; ++i)
	{
		for (int j = 0; j < output.getShape()[1]; ++j)
		{
			for (int k = 0; k < output.getShape()[2]; ++k)
			{
				output({i, j, k}) = input.block(
											 {i, j * strides, k * strides},
											 {1, poolSize, poolSize})
										.max();
			}
		}
	}
	return output;
}

syn::Tensor syn::Pooling::backward(const syn::Tensor &outputGrad)
{
	syn::Tensor inputGrad(input.getShape());
	for (int i = 0; i < output.getShape()[0]; ++i)
	{
		for (int j = 0; j < output.getShape()[1]; ++j)
		{
			for (int k = 0; k < output.getShape()[2]; ++k)
			{
				inputGrad({i, j * strides, k * strides}) = outputGrad({i, j, k});
			}
		}
	}
	return inputGrad;
}