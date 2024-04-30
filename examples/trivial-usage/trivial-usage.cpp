#include <Synapse/AI.h>
#include <iostream>


// This is an example about trivial usage of Synapse

int main() {
    // Define test inputs data
    syn::Data inputs({1, 1}, {
        {0}, {1}, {2},
    });

    // Define test labels data
    syn::Data labels({1, 1}, {
        {0}, {2}, {3},
    });

    // Let Synapse automatically create and teach new model based on created data
    syn::Model model = syn::create(inputs, labels);

    // Save the model
    model.save("model.txt");

    // Declare a very new model
    syn::Model model2;

    // Load a first model's data to it
    model2.load("model.txt");

    // Use the new model to predict some values and print the results
    std::cout << model2.predict(inputs[0]) << std::endl;
    std::cout << model2.predict(inputs[1]) << std::endl;
    std::cout << model2.predict(inputs[2]) << std::endl;

    return 0;
}