 !["Synapse Logo"](res/synapse-logo.png)

# Synapse — C++ Machine Learning Library

Synapse is a simple C++ machine learning library that fully automates processes of creating and learning machine learning models. The main aims of this project are to make C++ AI-development more simple and to stimulate emergence of new AI-startups and projects.

## How to build

To build Synapse, just navigate to its root folder and paste folowing commands:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

This will generate `Synapse` folder whith following subfolders:
* `bin` for dynamic library files
* `lib` for static library files
* `examples` for compiled examples

## How to use

As for other C++ modules, you need to do next steps to use Synapse:
* Include its headers (that, what is localed in Synapse's `include` folder)
* Link its binaries (these have `.so` or `.a` extentions on Linux and `.dll` or `.lib` extentions on Windows)

**Terminal:**

To compile Synapse-using project in terminal using gcc/g++, you can do this:

```bash
g++ -c main.cpp -I<synapse-install-path>/include
g++ main.o -o synapse-app -L<synapse-install-path>/lib -lsynapse
```

Change placeholder paths with real Synapse install path.

**CMake:**

On the other side, if you're using CMake, in your project's root folder you can create `include` and `lib` folders and respectevelly move there Synapce's headers and binaries.
`CMakeLists.txt` file should look something like this:

```cmake
cmake_minimum_required(VERSION 3.22)
project(SynapseProject)

set(CMAKE_CXX_STANDARD 17)

link_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(synapse-test main.cpp)

target_include_directories(synapse-test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(synapse-test PUBLIC synapse)
```

## Examples of code

Here is a simple example of using Synapse:

```c++
#include <Synapse/AI.h>

int main() {
  syn::Data inputs({2, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1});
  syn::Data labels({1, 1}, {0}, {1}, {1}, {0});

  syn::Model model({
    new syn::Dense(2, 3), new syn::Activation("leaky relu"),
    new syn::Dense(3, 1), new syn::Activation("sigmoid),
  });
  model.compile("MSE", "GD");

  model.train(30);
  model.save("models/model.txt");
}
```

In this example we are manualy creating learning data and model. You also can use special Synapse functionality, that will automaticaly generate a model with the most suitable architectire.

```c++
#include <Synapse/AI.h>

int main() {
  syn::Data inputs({2, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1});
  syn::Data labels({1, 1}, {0}, {1}, {1}, {0});

  syn::Model model(inputs, labels);

  model.save("models/model.txt");
}
```
## Available functionality

**syn::Model Interface Reference:**
| Method's name | Explaination |
| :------------- | :--- |
| compile | Set optimizer and loss function |
| predict | Push some input to model and get some output |
| train | Train model |
| evaluate | Get current loss value |

**syn::Data Interface Reference:**
| Method's name | Explaination |
| :------------- | :--- |
| shuffle | Shuffle data |
| size | Get size of data |
| merge | Merge two data instances to a signle one |
| extract | Get some part of a data |

**Layers:**
| Name | How to call in Synapse |
| :----- | :--- |
| Dense | syn::Dense(int nInput, int nOutput) |
| Activation | syn::Activation(const std::string& type)  |
| Convolutional | syn::Conv() |

**Activation functions:**
| Name | How to call in Synapse |
| :----- | :---: |
| ReLU | "relu"   |
| Leaky ReLU | "leaky relu"  |
| Sigmoid | "sigmoid"  |

**Loss functions:**
| Name | How to call in Synapse |
| :----- | :---: |
| Mean Squared Error | "MSE"   |
| Binary Cross-Entropy | "BCE"   |
| Categorical Cross-Entropy | "CCE"   |

**Optimizers:**
| Name | How to call in Synapse |
| :----- | :---: |
| Gradient Descent | "GD"   |
| Sticastic Gradient Descent | "SGD"   |

