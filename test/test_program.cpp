#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "Synapse/linear.h"
#include "Synapse/AI/model.h"
#include "Synapse/AI/layers.h"
#include "Synapse/AI/data.h"

void print(const std::vector<double>& vec) {
    for (int i = 0; i < vec.size() - 1; ++i) {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec.back() << '\n';
}

int main() {
    std::srand(std::time(nullptr));

    syn::Model model({
        new syn::Dense(1, 5),
        new syn::Activation("leaky relu"),
        new syn::Dense(5, 1),
    });
    model.compile("GD", "MSE");

    syn::Data training({1, 1}, {
        {-1}, {0}, {1}, {2},
    });

    syn::Data labels({1, 1}, {
        {-4}, {-2}, {2}, {4},
    });

    for (int i = 0; i < training.size(); ++i) {
        print(model.predict(training[i]).getData());
    }

    model.train(training, labels, 500, true);
    model.save("model.txt");

    syn::Model model1;
    model1.load("model.txt");
    model1.save("model1.txt");

    std::cout << '\n';
    for (int i = 0; i < training.size(); ++i) {
        print(model1.predict(training[i]).getData());
    }

    return 0;
}