#pragma once

namespace syn
{
    class IClonable
    {
    public:
        virtual IClonable *clone() const = 0;
    };
}