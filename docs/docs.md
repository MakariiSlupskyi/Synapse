# Documentation

This is a documentation page, where you can learn about all Synapse's most frequently utilized functionalities to be able to efficiently use it in your project.

## Automated model creation

Following code example shows the basic usage of Synapse

```c++
#include <Synapse/AI.h>
#include <iostream>

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
```

## Data processing

```c++
#include <Synapse/AI.h>

int main() {
    // Define training data
    syn::Data inputs({
        syn::Tensor({1, 6, 6}, {
            0, 1, 1, 1, 1, 0,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            0, 1, 1, 1, 1, 0,
        }), syn::Tensor({1, 6, 6}, {
            1, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            1, 0, 0, 0, 0, 1,
        }),
    });

    ml::Data labels({
        ml::Vector({1, 0}),
        ml::Vector({0, 1}),
    });

    // Save training data
    inputs.save("inputs.txt");
    labels.save("labels.txt");

    return 0;
}
```

## Models creation

```c++
#include <Synapse/AI.h>

int main()
{
    // FFNN
    syn::Model first({
        new syn::Dense(1, 5),
        new syn::Activation("leaky relu"),
        new syn::Dense(5, 1),
    });
    first.compile("GD", "MSE");

    // CNN
    ml::Model second({
        new syn::Convolutional({1, 6, 6}, {3, 3}, 4), // input shape = {1, 6, 6} , kernel shape = {3, 3} , depth = 4
        new syn::Pooling(2, 2), // strides = 2, pooling size = 2
        new syn::Activation("ReLU"),
        new syn::Flatten(),
        new syn::Dense(16, 2),
        new syn::Activation("sigmoid"),
    });
    second.compile("SGD", "MSE");
}
```
