#pragma once

#include <fstream>

namespace syn
{
    class ISavable
    {
    public:
        virtual void save(std::ofstream &file) const = 0;
        virtual void load(std::ifstream &file) = 0;
    };
}