#include "Synapse/AI/model.h"
#include "Synapse/AI/optimizers.h"
#include "Synapse/AI/layers.h"
#include "Synapse/AI/data.h"
#include "Synapse/AI/functions.h"
#include <cmath>

syn::Model::Model(const std::vector<syn::ILayer *> &layers)
	: layers(layers), lossType(""), optimType("")
{
}

syn::Model::Model(const syn::ModelBuilder &builder)
{
	layers.resize(builder.getNumLayer());
	for (int i = 0; i < builder.getNumLayer(); ++i)
	{
		layers[i] = builder.getLayers()[i];
	}

	this->lossType = builder.getLossType();
	this->optimType = builder.getOptimType();
}

syn::Model::Model(const syn::Model &other) // FIXME
{
	for (int i = 0; i < other.layers.size(); ++i)
	{
		layers.push_back(dynamic_cast<syn::ILayer *>(other.layers[i]->clone()));
	}

	// Copy strings by value (efficient)
	lossType = other.lossType;
	optimType = other.optimType;
}

void syn::Model::compile(const std::string &optimType, const std::string &lossType)
{
	this->optimType = optimType;
	this->lossType = lossType;
}

double syn::Model::evaluate(const syn::Data &inputs, const syn::Data &labels)
{
	syn::Tensor loss = labels[0] - this->predict(inputs[0]);
	loss.apply([](double x) -> double
			   { return x * x; });

	for (int i = 1; i < inputs.size(); ++i)
	{
		loss += (labels[i] - this->predict(inputs[i])).apply([](double x) -> double
															 { return x * x; });
	}
	return loss.sum();
}

syn::Tensor syn::Model::predict(const syn::Tensor &inputs)
{
	syn::Tensor output = inputs;
	for (int i = 0; i < layers.size(); ++i)
	{
		output = layers[i]->forward(output);
	}
	return output;
}

void syn::Model::train(const syn::Data &inputs, const syn::Data &labels, int epoches, bool printLoss)
{
	if (lossType == "" || optimType == "")
	{
		throw std::invalid_argument("Model is not compiled so it cannot be trained\n");
	}

	if (optimizer == nullptr)
	{
		optimizer = syn::optimizers.at(optimType)(this, &layers);
	}

	optimizer->train(inputs, labels, epoches, printLoss);
}

void syn::Model::backward(const syn::Tensor &loss)
{
	auto l = loss;
	for (int i = (int)layers.size() - 1; i >= 0; --i)
	{
		l = layers[i]->backward(l);
	}
}

void syn::Model::update(double rate)
{
	int size = layers.size();
	for (int i = size - 1; i >= 0; --i)
	{
		layers[i]->step(rate / size);
	}
}

void syn::Model::save(const std::string &path) const
{
	std::ofstream file(path);

	if (!file.is_open())
	{
		throw std::invalid_argument("Failed to open file " + path + "\n");
	}
	else if (lossType == "" || optimType == "")
	{
		throw std::invalid_argument("Model is not compiled so it cannot be saved");
	}

	this->save(file);
}

syn::Model &syn::Model::load(const std::string &path)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw std::invalid_argument("Can't open file: " + path);
	}

	this->load(file);

	return *this;
}

void syn::Model::randomize()
{
	for (int i = 0; i < layers.size(); ++i)
	{
		layers[i]->randomize();
	}
}

void syn::Model::tune(double alpha)
{
	for (int i = 0; i < layers.size(); ++i)
	{
		layers[i]->tune(alpha);
	}
}

syn::Model syn::Model::clone() const
{
	return syn::Model();
}

void syn::Model::save(std::ofstream &file) const
{
	file << optimType << ' ' << lossType << '\n';
	file << layers.size() << '\n';

	for (int i = 0; i < layers.size(); ++i)
	{
		layers[i]->save(file);
	}
}

void syn::Model::load(std::ifstream &file)
{
	std::string line;

	// read the number of layers
	std::getline(file, line);
	auto temp = syn::split(line);
	optimType = temp[0];
	lossType = temp[1];

	// read and create layers
	std::getline(file, line);
	int nLayer = std::stoi(line);

	// Read and create layers
	for (int i = 0; i < nLayer; ++i)
	{
		std::getline(file, line);
		layers.push_back(syn::layers.at(line)());
		layers.back()->load(file);
	}

	file.close();
}