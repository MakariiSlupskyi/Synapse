#include "Synapse/AI/data.h"
#include <stdexcept>

syn::Data::Data(syn::Tensor* data, int lenght) : data(data), lenght(lenght)
{}

syn::Data::Data(const std::vector<int>& shape, const std::vector<std::vector<double>>& vecData)
    : shape(shape), lenght(vecData.size())
{
    data = new syn::Tensor[vecData.size()];
    for (int i = 0; i < lenght; ++i) {
        data[i] = syn::Tensor(shape, vecData.at(i));
    }
}

syn::Data::Data(const std::vector<syn::Tensor>& data) : shape(data[0].getShape()), lenght(data.size())
{
    for (int i = 0; i < data.size() - 1; ++i) {
        if (data[i] != data[i + 1]) {
            throw std::invalid_argument("Invalid given data");
        }
    }
    this->data = new syn::Tensor[lenght];
    std::copy(data.begin(), data.end(), this->data);
}

syn::Tensor& syn::Data::operator[](int index) {
    return data[index];
}

syn::Tensor syn::Data::operator[](int index) const {
    return data[index];
}