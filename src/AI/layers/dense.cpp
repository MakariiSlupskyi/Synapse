#include "Synapse/AI/layers/dense.h"
#include "Synapse/AI/functions/other_funcs.h"

syn::Dense::Dense(int nInput, int nOutput)
    : input({nInput, 1}), output({nOutput, 1}),
      biases({nOutput, 1}), weights({nOutput, nInput}),
      outputGrad({nOutput, 1}), weightsGrad({nOutput, nInput})
{
    biases.randomize();
    weights.randomize();
}

void syn::Dense::randomize()
{
    biases.randomize();
    weights.randomize();
}

void syn::Dense::tune(double alpha)
{
    biases.tune();
    weights.tune();
}

void syn::Dense::save(std::ofstream &file) const
{
    file << "Dense\n";

    file << input.getShape()[0] << ' ' << output.getShape()[0] << '\n';

    for (auto elem : biases.getData())
    {
        file << elem << ' ';
    }
    file << '\n';

    for (auto elem : weights.getData())
    {
        file << elem << ' ';
    }
    file << '\n';
}

void syn::Dense::load(std::ifstream &file)
{
    std::string line;

    std::getline(file, line);

    std::vector<int> data = syn::splitToI(line);

    *this = syn::Dense(data[0], data[1]);

    std::getline(file, line);
    biases.fill(syn::splitToD(line));

    std::getline(file, line);
    weights.fill(syn::splitToD(line));
}

syn::Dense *syn::Dense::clone() const
{
    auto res = new syn::Dense(input.getShape()[0], output.getShape()[0]);

    res->biases = biases;
    res->weights = weights;

    return res;
}

syn::Tensor syn::Dense::forward(const syn::Tensor &input)
{
    this->input = input;
    output = weights.matMul(input);
    output += biases;
    return output;
}

syn::Tensor syn::Dense::backward(const syn::Tensor &outputGrad)
{
    weightsGrad = outputGrad.matMul(input.matTrans());

    this->outputGrad += outputGrad;
    this->weightsGrad += weightsGrad;

    return weights.matTrans().matMul(outputGrad);
}

void syn::Dense::step(double rate)
{
    weights -= weightsGrad * rate;
    biases -= outputGrad * rate;

    weightsGrad.zeros();
    outputGrad.zeros();
}