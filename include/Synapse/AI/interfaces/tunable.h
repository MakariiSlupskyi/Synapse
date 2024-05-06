#pragma once

namespace syn
{
    class ITunable
    {
    public:
        virtual void randomize() = 0;
        virtual void tune(double alpha) = 0;
    };
}