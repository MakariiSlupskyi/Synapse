#include "MLL/AI/Data.h"
#include <stdexcept>

ml::Data::Data(const std::vector<ml::Tensor>& data) : shape(data[0].getShape())
{
    for (int i = 0; i < data.size() - 1; ++i) {
        if (data[i] != data[i + 1]) {
            throw std::invalid_argument("Invalid data!");
        }
    }
}

ml::Data::Data(const std::vector<int>& shape, const std::vector<std::vector<double>>& data) 
    : shape(shape.size() == 1 ? std::vector<int>{shape[0], 1} : shape)
{
    this->data.reserve(data.size());
    for (int i = 0; i < data.size(); ++i) {
        this->data.emplace_back(shape, data[i]);
    }
}