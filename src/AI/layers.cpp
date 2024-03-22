#include "Synapse/AI/layers.h"

const std::map<std::string, std::function<syn::Layer*(std::ifstream& file)>> syn::file_layers {
	{ "Activation", [](std::ifstream& file) -> syn::Layer* { return new syn::Activation(file); } },
	{ "Convolutional", [](std::ifstream& file) -> syn::Layer* { return new syn::Convolutional(file); } },
	{ "Dense", [](std::ifstream& file) -> syn::Layer* { return new syn::Dense(file); } },
	{ "Flatten", [](std::ifstream& file) -> syn::Layer* { return new syn::Flatten(file); } },
	{ "Pooling", [](std::ifstream& file) -> syn::Layer* { return new syn::Pooling(file); } },
};