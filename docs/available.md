# Available Functionality

## syn::Model Interface Reference

| Method's Name | Explanation                                      |
| :------------ | :----------------------------------------------- |
| `compile`     | Set optimizer and loss function                  |
| `predict`     | Push some input to the model and get some output |
| `train`       | Train the model                                  |
| `evaluate`    | Get the current loss value                       |

## syn::Data Interface Reference

| Method's Name | Explanation                                |
| :------------ | :----------------------------------------- |
| `shuffle`     | Shuffle data                               |
| `size`        | Get the size of the data                   |
| `merge`       | Merge two data instances into a single one |
| `extract`     | Get a part of the data                     |

## Layers

| Name          | How to Call in Synapse                     |
| :------------ | :----------------------------------------- |
| Dense         | `syn::Dense(int nInput, int nOutput)`      |
| Activation    | `syn::Activation(const std::string& type)` |
| Convolutional | `syn::Conv()`                              |

## Activation Functions

| Name       | How to Call in Synapse |
| :--------- | :--------------------: |
| ReLU       |        `"relu"`        |
| Leaky ReLU |     `"leaky relu"`     |
| Sigmoid    |      `"sigmoid"`       |

## Loss Functions

| Name                      | How to Call in Synapse |
| :------------------------ | :--------------------: |
| Mean Squared Error        |        `"MSE"`         |
| Binary Cross-Entropy      |        `"BCE"`         |
| Categorical Cross-Entropy |        `"CCE"`         |

## Optimizers

| Name                        | How to Call in Synapse |
| :-------------------------- | :--------------------: |
| Gradient Descent            |         `"GD"`         |
| Stochastic Gradient Descent |        `"SGD"`         |
