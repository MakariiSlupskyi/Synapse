# Available functionality

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
| Activation | syn::Activation(const std::string& type) |
| Convolutional | syn::Conv() |

**Activation functions:**
| Name | How to call in Synapse |
| :----- | :---: |
| ReLU | "relu" |
| Leaky ReLU | "leaky relu" |
| Sigmoid | "sigmoid" |

**Loss functions:**
| Name | How to call in Synapse |
| :----- | :---: |
| Mean Squared Error | "MSE" |
| Binary Cross-Entropy | "BCE" |
| Categorical Cross-Entropy | "CCE" |

**Optimizers:**
| Name | How to call in Synapse |
| :----- | :---: |
| Gradient Descent | "GD" |
| Sticastic Gradient Descent | "SGD" |
