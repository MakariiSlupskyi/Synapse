#pragma once

#include "Synapse/AI/data.h"

namespace syn {
    class Dataset {
    public:
        Dataset(const syn::Data& inputs, const syn::Data& outputs);

        void setInputs(const syn::Data& other);
        void setOutputs(const syn::Data& other);

        syn::Data getInputs() const { return inputs; }
        syn::Data getOutputs() const { return outputs; }

		int size() const { return lenght; };
        
		syn::Dataset& shuffle();

		syn::Dataset merge(const syn::Dataset& other);
		syn::Dataset extract(int start, int size) const;

    private:
        syn::Data inputs, outputs;
        int lenght;
    };
}
