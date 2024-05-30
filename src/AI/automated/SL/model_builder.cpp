#include "Synapse/AI/automated/SL/model_builder.h"
#include "Synapse/AI/layers.h"
#include "Synapse/AI/model.h"

syn::ModelBuilder::ModelBuilder(const syn::Data &inputs, const syn::Data &labels)
{
    int nInputDim = inputs[0].getShape().size();
    int nOutputDim = labels[0].getShape().size();

    if (nInputDim == 2 && nOutputDim == 2)
        *this = syn::FFNNBuilder(inputs, labels); // Delegate to FFNN builder
    else if (nInputDim == 3 && nOutputDim == 2)
        *this = syn::CNNBuilder(inputs, labels); // Delegate to CNN builder
    else
        throw std::runtime_error("Unfortunately, Synapse cannot create a model based on given training data");
}

syn::Model syn::ModelBuilder::build() const
{
    return syn::Model(*this);
}

syn::FFNNBuilder::FFNNBuilder(const syn::Data &inputs, const syn::Data &labels)
{
    // This is a temperary functionality and will be developped in the future

    int nInput = inputs[0].getData().size();
    int nOutput = labels[0].getData().size();

    // Defining architecture-defining variables
    int nBaseNeuron = std::max(nInput, nOutput) + 5;
    int nDenseLayer = std::max(nInput, nOutput) + 5;
    nLayer = nDenseLayer * 2;

    // Generate layers
    int nCur = -1;  // Number of neurons in current layer
    int nNext = -1; // Number of neurons in next layer

    layers = new syn::ILayer *[nLayer];
    for (int i = 0; i < nLayer; i += 2)
    {
        // Calculation numbers of neurons in current and next layers
        nCur = (i == 0) ? nInput : nBaseNeuron;
        nNext = (i == nLayer - 2) ? nOutput : nBaseNeuron;

        // Adding layers
        layers[i] = new syn::Dense(nCur, nNext);
        layers[i + 1] = new syn::Activation("leaky relu");
    }

    // Defifing types of optimizer and loss function
    lossType = "MSE";
    optimType = "GD";
}

syn::CNNBuilder::CNNBuilder(const syn::Data &inputs, const syn::Data &labels)
{
}