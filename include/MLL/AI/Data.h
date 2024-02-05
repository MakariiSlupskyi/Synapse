#pragma once

#include "MLL/Linear/Tensor.h"

namespace ml {
    class Data
    {
    public:
        Data(const std::vector<ml::Tensor>& data);
        Data(const std::vector<int>& shape, const std::vector<std::vector<double>>& data);
        
		int size() const { return data.size(); };
        
		Data merge(const ml::Data& other);
		Data extract(int start, int size) const;

		ml::Tensor operator[](int index) const;

    private:
        std::vector<int> shape;
        std::vector<ml::Tensor> data;
    };
}