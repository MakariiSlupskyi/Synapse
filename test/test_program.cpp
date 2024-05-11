#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "Synapse/linear.h"
#include "Synapse/AI/model.h"
#include "Synapse/AI/layers.h"
#include "Synapse/AI/data.h"

#include "Synapse/AI/automated.h"

void print(const std::vector<double> &vec)
{
    for (int i = 0; i < vec.size() - 1; ++i)
    {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec.back() << '\n';
}

class Entity;

int main()
{
    std::srand(std::time(nullptr));

    // // Test 1
    // syn::Model model({
    //     new syn::Dense(1, 5),
    //     new syn::Activation("leaky relu"),
    //     new syn::Dense(5, 1),
    // });
    // model.compile("GD", "MSE");

    // syn::Data training({1, 1}, {
    //     {-1}, {0}, {1}, {2},
    // });

    // syn::Data labels({1, 1}, {
    //     {-4}, {-2}, {2}, {4},
    // });

    // for (int i = 0; i < training.size(); ++i) {
    //     print(model.predict(training[i]).getData());
    // }

    // model.train(training, labels, 500, true);
    // model.save("model.txt");

    // syn::Model model1;
    // model1.load("model.txt");
    // model1.save("model1.txt");

    // std::cout << '\n';
    // for (int i = 0; i < training.size(); ++i) {
    //     print(model1.predict(training[i]).getData());
    // }

    // Test 3
    std::srand(std::time(nullptr));

    // syn::Model model({
    //     new syn::Dense(1, 5),
    //     new syn::Activation("leaky relu"),
    //     new syn::Dense(5, 1),
    // });

    // syn::Tensor inputs({1, 1}, {0.13});

    // syn::Population<Entity> population(model, 5);

    // for (int i = 0; i < 100; ++i)
    // {
    //     population.run(inputs);
    //     std::cout << "\n";
    // }

    syn::Data data({1, 2}, {{1, 0}, {0, 1}, {0, 0}, {1, 1}});
    data.save("daat.txt");

    std::cout << "Hooray!!\n";

    return 0;
}

class Entity : public syn::Agent
{
public:
    Entity(const syn::Model &model) : Agent(model)
    {
    }

    Entity(const syn::Model &model, std::initializer_list<double> list) : Agent(model)
    {
        x = *list.begin();
    }

    void step(const syn::Tensor &inputs) override
    {
        x += this->getOutput(inputs)({0, 0});
        if (x < 0)
        {
            this->kill();
        }
        std::cout << x << ' ';
    }

private:
    double x;
};