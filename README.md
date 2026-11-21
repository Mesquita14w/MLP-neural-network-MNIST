# A Multilayer Perceptron Neural Network Implementation in Numpy

This repository contains a MLP implementation codes written by me to apply the knowledges that i learned in the Artificial Neural Networks course at my college, i will just use Math and NN Theory and the Numpy to build this project. All these codes were written with the aim of learning and practicing, so i accept suggestions for improvement and corrections in case of errors or mistakes. I really hope you enjoy, Cheers!

---

## The Problem and the Architecture behind the MLP

I choose the Digit Recognizer problem with the MNIST dataset because it's a famous problem and i believe this the perfect problem to test your skills in the Neural Networks theory and if you really understand all the theory and techniques to how to resolve complex problems with Neural Networks. I opted to resolve this problem with a MLP because this architecture it served as the conceptual foundation for everything, representing the first model of an artificial neuron and multilayer feedforward networks trained via backpropagation.

## Math Theory and Network Architecture behind the Code

### Math Modeling:

Each MNIST image has $28 \times 28 = 784$ pixels. Each image is flattened into a column vector $x \in \mathbb{R}^{784}$ (normalized to $[0, 1]$ by dividing by $255.0$) and each label $y \in \{0, \dots, 9\}$ is represented as a one-hot vector $Y \in \mathbb{R}^{10}$.

Forward Propagation (for a batch of size $m$):

- Let $X \in \mathbb{R}^{n_0 \times m}$ be the input matrix (where $n_0 = 784$ and each column is a sample). For layer $l \in \{1, \dots, L\}$:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$

- where $A^{[0]} = X$, $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$, and $b^{[l]} \in \mathbb{R}^{n_l \times 1}$ (added via broadcasting across the $m$ columns).

* **Hidden layers ($l < L$):** Use **ReLU**, which helps avoid rapid gradient attenuation in positive regions:
  $$g(z) = \max(0, z), \quad g'(z) = \begin{cases} 1, & z > 0 \\ 0, & z \le 0 \end{cases}$$

* **Output layer ($l = L$):** Use **Softmax** to convert logits $Z^{[L]}$ into class probabilities:
  $$\hat{y}_{i,j} = \text{softmax}(z^{[L]})_{i,j} = \frac{e^{z^{[L]}_{i,j} - \max_k z^{[L]}_{k,j}}}{\sum_{k=1}^{10} e^{z^{[L]}_{k,j} - \max_k z^{[L]}_{k,j}}}$$
  _(Subtracting $\max_k z_{k,j}$ improves numerical stability and prevents `NaN` values from exponential overflow.)_

Loss Function (Categorical Cross-Entropy):

- For a mini-batch of $m$ examples and target matrix $Y \in \mathbb{R}^{10 \times m}$:

$$\mathcal{L} = -\frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{10} Y_{i,j} \ln(\hat{Y}_{i,j} + \varepsilon)$$

Backpropagation (Chain Rule in Matrix Form):

- A major mathematical advantage of combining **Softmax + Cross-Entropy** is the clean simplification of the gradient at the output layer:

$$dZ^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{1}{m}(\hat{Y} - Y)$$

- For any layer $l$ (moving backward from $L$ to $1$):

$$dW^{[l]} = dZ^{[l]} (A^{[l-1]})^\top$$

$$db^{[l]} = \sum_{j=1}^{m} dZ^{[l]}_{:, j}$$

And to propagate the error to the previous layer ($l - 1$):

$$dA^{[l-1]} = (W^{[l]})^\top dZ^{[l]}$$

$$dZ^{[l-1]} = dA^{[l-1]} \odot g'^{[l-1]}(Z^{[l-1]})$$

where $\odot$ denotes the Hadamard (element-wise) product.

### Network Architecture and Initialization:

A strong, fast-training baseline architecture for MNIST is:

- **Input:** $784$ neurons
- **Hidden Layer 1:** $128$ neurons + **ReLU**
- **Hidden Layer 2:** $64$ neurons + **ReLU**
- **Output:** $10$ neurons + **Softmax**

**Why not initialize weights with zeros or large numbers?**

- If all weights are identical (such as all zeros), every neuron in a layer computes the same activation and receives the same gradient—resulting in **symmetry** where the network fails to learn diverse features. With **He Initialization** (well-suited for ReLU):

$$W^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n_{l-1}}\right)$$

This keeps activation variances within a healthy scale across layers.

> **Numerical verification tip:** Before fully trusting your backward pass, test it on a small batch using **Gradient Checking** via finite differences $(f(\theta + h) - f(\theta - h))/(2h)$ with $h \approx 10^{-7}$. The relative error should be around $10^{-7}$ or lower, confirming that your matrix derivatives are exact.

### Metrics and Evaluation Beyond Accuracy:

- On MNIST, a well-tuned MLP without convolutions typically achieves around **97% to 98.5%** overall accuracy. Even so, computing per-class metrics helps reveal where the model confuses digits (such as `4` vs. `9`, `3` vs. `8`, or `5` vs. `3`):

* **Confusion Matrix ($10 \times 10$):** Rows represent the true class $y$ and columns represent the predicted class $\hat{y}$, with the main diagonal showing correct predictions.
* **Precision per class $c$:** $\frac{TP_c}{TP_c + FP_c}$ — when the model predicts digit $c$, how often is it right?
* **Recall per class $c$:** $\frac{TP_c}{TP_c + FN_c}$ — out of all actual instances of digit $c$, what fraction did the model find?
* **F1-Score:** The harmonic mean of precision and recall, useful for summarizing performance per digit or handling class imbalance.

## Code and Files Structure:

```
data/train.csv
```

```
data/test.csv
```

- Train and Test data files (downloaded on the kaggle page of the project, link below).
- https://www.kaggle.com/competitions/digit-recognizer

```
analysis.py
```

- The file i used to do the Data Analysis of the Datasets.
- I do not used a jupyter notebook with all my tests, because i don't liked use this in my personal projects.

```
mlp.py
```

- The code of the model. All the architecture and math functions used in the code.
- All the metrics of the model is in the file `metrics.png`.
