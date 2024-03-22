#include "Synapse/AI/model.h"
#include "Synapse/AI/optimizers.h"
#include "Synapse/AI/layers.h"
#include "Synapse/AI/functions.h"

syn::Model::Model(const std::vector<syn::Layer*>& layers)
: layers(layers), optimizer(nullptr), lossType("none")
{}

syn::Model::~Model() {
	for (int i = 0; i < layers.size(); ++i) {
		delete layers[i];
	}
	delete optimizer;
}

void syn::Model::compile(const std::string& optimType, const std::string& lossType) {
	this->optimType = optimType;
	this->lossType = lossType;
}

double syn::Model::evaluate(const syn::Data& inputs, const syn::Data& labels) {
    syn::Tensor loss = (labels[0] - this->predict(inputs[0])).apply([](double x) -> double {
		return x * x;
	});
	for (int i = 1; i < inputs.size(); ++i) {
		loss += (labels[i] - this->predict(inputs[i])).apply([](double x) -> double {
			return x * x;
		});
	}
	return loss.sum();
}

syn::Tensor syn::Model::predict(const syn::Tensor& inputs) {
	syn::Tensor output = inputs;
	for (int i = 0; i < layers.size(); ++i) {
		output = layers[i]->forward(output);
	}
	return output; 
}

void syn::Model::train(const syn::Data& inputs, const syn::Data& labels, int epoches) {
	if (optimizer == nullptr) {
		optimizer = syn::optimizers.at(optimType)(this, &layers);
	}
	
	optimizer->train(inputs, labels, epoches);
}

void syn::Model::save(const std::string& path) const {
	std::ofstream file(path);

	if (!file.is_open()) {
		throw std::invalid_argument("Failed to open file " + path + "\n");
	} else if (lossType == "" || optimType == "") {
		throw std::invalid_argument("Model can't be saved, because it's not compiled");
	}

	file << optimType << ' ' << lossType << '\n';
	file << layers.size() << '\n';
	
	for (int i = 0; i < layers.size(); ++i) {
		layers[i]->write(file);
	}
}

syn::Model& syn::Model::load(const std::string& path) {
	std::ifstream file(path);

	if (!file.is_open()) {
		throw std::invalid_argument("Can't open file: " + path);
	}
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
	for (int i = 0; i < nLayer; ++i) {
		std::getline(file, line);
		layers.push_back(syn::file_layers.at(line)(file));
	}

	file.close();
	
	return *this;
}

void syn::Model::backward(const syn::Tensor& loss) {
	auto l = loss;
	for (int i = (int)layers.size() - 1; i >= 0; --i) {
		l = layers[i]->backward(l);
	}
}

void syn::Model::update(double rate) {
	int size = layers.size();
	for (int i = size - 1; i >= 0; --i) {
		layers[i]->update(rate / size);
		layers[i]->clearGradient();
	}
}