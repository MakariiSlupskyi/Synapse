# Welcome to Synapse

<center>
    <img width=500 style="border-radius: 30px" src=res/synapse-logo.png>
</center>

## Introduction

<b>Synapse</b> is a simple C++ machine learning library designed to <b>automate</b> various aspects of the machine learning pipeline, from data preprocessing to model training.
It provides a convenient interface for developers to integrate machine learning algorithms into their applications effortlessly.

## Installation

You can find Synapse's releases and source code on its [GitHub](https://github.com/MakariiSlupskyi/Synapse) page.

To build Synapse on you own, navigate to its root directory and paste following commands:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

This will generate a seperate `Synapse` directory with built binaries and examples. If you don't want to build examples, add this when building project:

```bash
cmake -DBUILD_EXAMPLES=False ..
cmake --build .
```

## Quick Start

To check if Synapse works properly use this simple code example

```c++
#include <Synapse/AI.h>

int main()
{
    syn::Data inputs({1, 1}, {{0}, {1}, {2}});
    syn::Data labels({1, 1}, {{0}, {2}, {3}});

    syn::Model model = syn::create(inputs, labels);
    model.save("model.txt");
}
```
