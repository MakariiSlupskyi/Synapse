#pragma once

#include "Synapse/AI/model.h"

namespace syn {
    class Entity {
    public:
        Entity(const syn::Model& agent);

        // Proceed one step in environment
        virtual void action() = 0;

        // Give agent a reward or punishment
        virtual void reward(double reward) = 0;

        // Randomize agent's parameters
        void forget();

        // Slightly change agent's parameters
        void mutate(double alpha = 0.1);

    protected:
        syn::Model agent;

    private:
    };
}