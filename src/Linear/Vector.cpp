#include "Synapse/linear/vector.h"

syn::Tensor syn::Vector(int size) {
    return syn::Tensor({size, 1});
}

syn::Tensor syn::Vector(int size, const std::vector<double>& data) {
    return syn::Tensor({size, 1}, data);
}

syn::Tensor syn::Vector(const std::vector<double>& data) {
    return syn::Tensor({(int)data.size(), 1}, data);
}