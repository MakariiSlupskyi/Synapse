#include "Synapse/linear/tensor_funcs.h"
#include <cmath>

syn::Tensor syn::abs(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return std::abs(x); });
}

syn::Tensor syn::square(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return x * x; });
}

syn::Tensor syn::log(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return std::log(x); });
}

syn::Tensor syn::exp(const syn::Tensor& tensor) {
    return tensor.apply([](double x) -> double { return std::exp(x); });
}

syn::Tensor syn::reverse(const syn::Tensor& tensor) {
    return syn::Tensor(tensor).reverse();
}