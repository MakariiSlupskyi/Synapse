#include "Synapse/AI/automated/SL/create.h"
#include "Synapse/AI/layers.h"
#include <stdexcept>
#include <vector>

//////////////////////////////////////////////////////////////////
//
//  This is a demo implementation and will be further developed !!!
//
//////////////////////////////////////////////////////////////////

/// @brief Function, that automatically generates and trains models by training data
syn::Model syn::create(const syn::Data &inputs, const syn::Data &labels)
{
    // Variable that defines a level of creation precision
    const int n = 5;

    std::vector<syn::Model> models;
    models.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        models.emplace_back(syn::ModelBuilder(inputs, labels));
    }

    auto res = syn::Model(syn::ModelBuilder(inputs, labels));
    res.train(inputs, labels, 1000);

    return res;
}