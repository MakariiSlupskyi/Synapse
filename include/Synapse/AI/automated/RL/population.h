#pragma once

#include "Synapse/AI/automated/RL/agent.h"
#include <vector>
#include <iostream>
#include <memory>

namespace syn
{
    template <typename T>
    class Population
    {
    public:
        Population(const syn::Model &baseModel, int size);

        int getCurSize() const { return population.size(); }

        void create(std::initializer_list<double> list);

        // Make every agent proceed one step of simulation
        void step(const syn::Tensor &inputs);

        void run(const syn::Tensor &inputs);

    private:
        int size, genNumber;
        syn::Model baseModel;
        std::unique_ptr<syn::Agent> best;
        std::vector<std::unique_ptr<syn::Agent>> population;
    };
}

template <typename T>
syn::Population<T>::Population(const syn::Model &model, int size)
    : size(size), genNumber(-1), baseModel(model)
{
    best = std::make_unique<T>(model);
}

template <typename T>
void syn::Population<T>::create(std::initializer_list<double> list)
{
    population.clear();
    population.reserve(size);

    if (genNumber == -1) // If creation first generation
    {
        for (int i = 0; i < size; ++i)
        {
            population.push_back(std::make_unique<T>(baseModel, list));
            population[i]->forget();
        }
    }
    else
    {
        for (int i = 0; i < size; ++i)
        {
            population.push_back(std::make_unique<T>(baseModel, list));
            population[i]->mutate(0.1);
        }
    }

    ++genNumber;
}

template <typename T>
void syn::Population<T>::step(const syn::Tensor &inputs)
{
    for (int i = 0; i < population.size(); ++i)
    {
        population[i]->step(inputs);

        if (population[i]->getScore() > best->getScore())
        {
            std::swap(population[i], best);
        }

        // Delete dead agents
        if (!population[i]->isAlive())
        {
            population[i] = std::move(population.back());
            population.pop_back();
        }
    }
}

template <typename T>
void syn::Population<T>::run(const syn::Tensor &inputs)
{
    if (population.size() == 0)
    {
        this->create({0.1});
    }
    else
    {
        this->step(inputs);
    }
}