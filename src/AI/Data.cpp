#include "Synapse/AI/data.h"
#include <algorithm>
#include <cstdlib>
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
            throw std::invalid_argument("Invalid given data when creating an object of data");
        }
    }
    this->data = new syn::Tensor[lenght];
    std::copy(data.begin(), data.end(), this->data);
}

syn::Data::~Data() {
    delete[] data;
}

syn::Data& syn::Data::shuffle(int seed) {
    if (seed != -1) { std::srand(seed); }  // Set seed if it is not default

    for (int i = 0; i < lenght; ++i) {
        int j = std::rand() % (lenght - i) + i;
        std::swap(data[i], data[j]);
    }
    return *this; 
}

syn::Data syn::Data::merge(const syn::Data& other) {
    if (shape.size() != other.size()) {
        throw std::runtime_error("Invalid given data for merging with it");
    } else {
        for (int i = 0; i < shape.size(); ++i) {
            if (shape[i] != other.shape[i]) {
                throw std::runtime_error("Invalid given data for merging with it");
            }
        }
    }
    auto resData = new syn::Tensor[lenght + other.lenght];
    std::copy(data, data + lenght, resData);
    std::copy(other.data, other.data + other.lenght, resData + lenght);

    return syn::Data(resData, lenght + other.lenght);
}

syn::Data syn::Data::extract(int start, int size) const {
    return *this;
}

syn::Data& syn::Data::operator=(const syn::Data& other) {
    delete[] data;

    shape = other.shape;
    data = other.data;
    lenght = other.lenght;

    return *this;
}