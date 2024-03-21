#pragma once

#include "Synapse/linear/tensor.h"
#include <vector>

namespace syn {
    class Data
    {
    public:
        Data(syn::Tensor* data, int lenght);
        Data(const std::vector<int>& shape, const std::vector<std::vector<double>>& vecData);
        Data(const std::vector<syn::Tensor>& data);
        ~Data() { delete[] data; }

		int size() const { return lenght; };
        
		Data merge(const syn::Data& other);
		Data extract(int start, int size) const;

		syn::Tensor& operator[](int index);
        syn::Tensor operator[](int index) const;

    private:
        std::vector<int> shape;
        syn::Tensor* data;
        int lenght;
    };

    class Dataset {
    public:
        Dataset(int size);
        Dataset(const syn::Data& inputData, const syn::Data& outputData);

    private:
        syn::Tensor* inputData, outputData;
    };
}