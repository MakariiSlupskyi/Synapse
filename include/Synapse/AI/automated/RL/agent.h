#pragma once

#include "Synapse/AI/model.h"

namespace syn
{
    class Agent
    {
    public:
        Agent(const syn::Model &model);
        Agent(const Agent *other);

        double getScore() const { return score; }
        bool isAlive() const { return alive; }

        syn::Tensor getOutput(const syn::Tensor &inputs);

        // Proceed one step in environment
        virtual void step(const syn::Tensor &inputs) = 0;

        // Give agent a reward or punishment
        void reward(double value);

        // Randomize agent's parameters
        void forget();

        // Slightly change agent's parameters
        void mutate(double alpha = 0.1);

        void kill() { alive = false; }

    protected:
        bool alive;
        syn::Model model;

    private:
        double score;
    };
}