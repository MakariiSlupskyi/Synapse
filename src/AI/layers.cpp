#include "Synapse/AI/layers.h"

const std::map<std::string, std::function<syn::ILayer *()>> syn::layers{
	{"Activation", []() -> syn::ILayer *
	 { return new syn::Activation(); }},
	{"Convolutional", []() -> syn::ILayer *
	 { return new syn::Convolutional(); }},
	{"Dense", []() -> syn::ILayer *
	 { return new syn::Dense(); }},
	{"Flatten", []() -> syn::ILayer *
	 { return new syn::Flatten(); }},
	{"Pooling", []() -> syn::ILayer *
	 { return new syn::Pooling(); }},
};