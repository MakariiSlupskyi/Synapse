#include "Synapse/AI/functions/activ_funcs.h"
#include <cmath>

syn::Tensor syn::relu(const syn::Tensor &tensor)
{
	return tensor.apply([](double x) -> double
						{ return std::fmax(0.0f, x); });
}

syn::Tensor syn::reluDeriv(const syn::Tensor &tensor)
{
	return tensor.apply([](double x) -> double
						{ return (x > 0.0f); });
}

syn::Tensor syn::leakyRelu(const syn::Tensor &tensor)
{
	return tensor.apply([](double x) -> double
						{ return x > 0.0f ? x : 0.1f * x; });
}

syn::Tensor syn::leakyReluDeriv(const syn::Tensor &tensor)
{
	return tensor.apply([](double x) -> double
						{ return x > 0.0f ? 1.0f : 0.1f; });
}

syn::Tensor syn::sigmoid(const syn::Tensor &tensor)
{
	return tensor.apply([](double x) -> double
						{ return x / 2.0f / (1.0f + std::fabs(x)) + 0.5f; });
}

syn::Tensor syn::sigmoidDeriv(const syn::Tensor &tensor)
{
	return tensor.apply([](double x) -> double
						{
		double t = 1.0f + std::fabs(x);
		return 0.5f / (t * t); });
}