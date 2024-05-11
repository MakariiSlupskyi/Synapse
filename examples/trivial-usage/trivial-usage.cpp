#include <Synapse/AI.h>
#include <iostream>

// This is an example about trivial usage of Synapse

int main()
{
    // Prepare training data
    syn::Data inputs({1, 1}, {{0}, {1}, {2}});
    syn::Data labels({1, 1}, {{0}, {2}, {3}});

    // Create and train new model!
    syn::Model model = syn::create(inputs, labels);

    // Save the model
    model.save("model.txt");

    // Declare a very new model
    syn::Model newModel;

    // Load a first model's data to it
    newModel.load("model.txt");

    // Use the new model to predict some values and print the results
    std::cout << newModel.predict(inputs[0]) << std::endl;
    std::cout << newModel.predict(inputs[1]) << std::endl;
    std::cout << newModel.predict(inputs[2]) << std::endl;

    return 0;
}