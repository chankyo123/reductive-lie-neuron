# EQUIVARIANT NEURAL NETWORKS FOR GENERAL LINEAR SYMMETRIES ON LIE ALGEBRAS

**Chankyo Kim¹*, Sicheng Zhao*, Minghan Zhu¹², Tzu-Yuan Lin³, Maani Ghaffari¹**
<br>
¹University of Michigan, ²University of Pennsylvania, ³Massachusetts Institute of Technology
<br>
*Equal contribution.

[Project Website](https://your-username.github.io/reductive-lie-neuron-page) | [Paper on ArXiv](https://arxiv.org/abs/your-paper-id) | [Video](https://youtu.be/your-video-id)

---

## About

Encoding symmetries is a powerful inductive bias for improving the generalization of deep neural networks. However, most existing equivariant models are limited to simple symmetries like rotations, failing to address the broader class of general linear transformations, GL(n), that appear in many scientific domains. We introduce **Reductive Lie Neurons (ReLNs)**, a novel neural network architecture exactly equivariant to these general linear symmetries.

![Applications of ReLN across various scientific domains](figures/fig1.pdf)
*ReLNs are applicable to a wide range of scientific domains governed by diverse Lie group symmetries, from physics and robotics to computer vision.*

Unlike previous methods like [LieNeurons](https://github.com/UMich-CURLY/LieNeurons), which are tailored for semi-simple Lie algebras (e.g., `so(3)`), our work introduces a general approach to construct **non-degenerate bilinear forms for any `n x n` matrix Lie algebra**, including reductive ones like `gl(n)`. This allows for the principled design of equivariant layers and nonlinearities for a much broader class of symmetries.

This repository provides the official code to reproduce the experiments in our paper, including algebraic benchmarks and a challenging drone state estimation task.

---

## Core Concept: Adjoint Equivariance by Design

A key contribution of our work is a unified framework that embeds diverse geometric inputs (like vectors and covariance matrices) into a common Lie algebra, where they transform consistently under the **adjoint action**. Our network is designed to commute with this action, guaranteeing equivariance.

![Equivariance Diagram](figures/fig3.pdf)
*Our network `f` is provably equivariant. A transformation `Ad_g` on the input results in the same transformation `Ad_g` on the output feature.*

To achieve this for general reductive algebras like `gl(n)`, we introduce a non-degenerate, Ad-invariant bilinear form:

`B(X, Y) = 2n * tr(XY) - tr(X)tr(Y)`

This form is the fundamental tool used to build our equivariant layers. Here’s a simple code snippet demonstrating how it creates an invariant feature:

```python
# From core/layers.py
import torch

class LNInvariant(nn.Module):
    """
    Computes an invariant scalar feature from a Lie algebra element
    using our non-degenerate bilinear form.
    """
    def __init__(self, in_channels, algebra_type='gl3'):
        super(LNInvariant, self).__init__()
        self.hat_layer = HatLayer(algebra_type) # Maps vector to matrix
        self.algebra_type = algebra_type

    def forward(self, x):
        """
        Input x: Lie algebra vectors
        Output: Invariant scalars
        """
        # 1. Map vector representation to matrix representation
        x_hat = self.hat_layer(x)

        # 2. Compute the invariant using the bilinear form B(X, X)
        invariant_scalar = killingform(x_hat, x_hat, self.algebra_type)

        # 3. Aggregate features (e.g., via mean)
        return invariant_scalar.mean(dim=[-2, -1])
```

For a more detailed interactive example, please see our [Toy Problem Notebook](examples/toy_problem.ipynb).

---

## Installation

To set up the environment, please follow these steps:

```bash
# Clone the repository
git clone [https://github.com/your-username/reductive-lie-neuron.git](https://github.com/your-username/reductive-lie-neuron.git)
cd reductive-lie-neuron

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Paper Results

All experiment scripts are located in the `experiments/` directory.

### Algebraic Benchmark: Platonic Solid Classification
This experiment validates `SL(3)` equivariance.
```bash
python experiments/platonic_solid/train.py --config configs/sl3_config.yaml
```

### Drone State Estimation
This experiment demonstrates uncertainty-aware `SO(3)` equivariance on velocity and covariance data.
```bash
python experiments/drone_estimation/train.py --config configs/drone_config.yaml
```

*Tip: We found that a lower learning rate (e.g., `1e-4` to `3e-5`) and careful weight initialization are crucial for stable training, especially for deeper networks.*

---

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{kim2025reln,
  title={{EQUIVARIANT NEURAL NETWORKS FOR GENERAL LINEAR SYMMETRIES ON LIE ALGEBRAS}},
  author={Kim, Chankyo and Zhao, Sicheng and Zhu, Minghan and Lin, Tzu-Yuan and Ghaffari, Maani},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```