#include "Synapse/AI/automated/RL/entity.h"

syn::Entity::Entity(const syn::Model& agent) : agent(agent)
{}

void syn::Entity::forget() {
    agent.randomize();
}

void syn::Entity::mutate(double alpha) {
    agent.tune(alpha);
}