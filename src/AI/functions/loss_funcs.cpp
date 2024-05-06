#include "Synapse/AI/functions/loss_funcs.h"

syn::Tensor syn::MSE(const syn::Tensor &predicted, const syn::Tensor &desired)
{
    return (predicted - desired).square();
}

syn::Tensor syn::MSEDeriv(const syn::Tensor &predicted, const syn::Tensor &desired)
{
    return (predicted - desired) * 2;
}
