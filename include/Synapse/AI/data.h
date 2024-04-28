#pragma once

#include "Synapse/linear/tensor.h"
#include <vector>

namespace syn {
    class Data
    {
    public:
        Data() = delete;
        Data(syn::Tensor* data, int lenght);
        Data(const std::vector<int>& shape, const std::vector<std::vector<double>>& vecData);
        Data(const std::vector<syn::Tensor>& data);
        ~Data();

		int size() const { return lenght; };
        
		Data& shuffle(int seed = -1);

		Data merge(const syn::Data& other);
		Data extract(int start, int size) const;

		syn::Tensor& operator[](int index) { return data[index]; }
        syn::Tensor operator[](int index) const { return data[index]; }

        syn::Data& operator=(const syn::Data& other);

    private:
        std::vector<int> shape;
        syn::Tensor* data;
        int lenght;
    };
}