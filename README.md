# Reductive Lie Neurons (ReLN)

**Official PyTorch implementation of "Reductive Lie Neurons" (ICLR 2025, Under Review)**

[Project Website](https://your-username.github.io/reductive-lie-neuron-page) | [Paper on ArXiv](https://arxiv.org/abs/your-paper-id) | [Video](https://youtu.be/your-video-id)

![Teaser Image or GIF of your best result](figures/teaser.gif)
*A brief, exciting one-sentence description of what this image/gif shows.*

---

## About

**Reductive Lie Neurons (ReLN)** is a novel equivariant neural network framework designed for Lie groups that are not necessarily semi-simple. Unlike previous methods like [LieNeurons](https://github.com/UMich-CURLY/LieNeurons), which are tailored for semi-simple Lie algebras with well-defined Killing forms (e.g., `so(3)`), our work introduces a general approach to construct **non-degenerate bilinear forms for any `n x n` matrix Lie algebra**, including reductive ones like `gl(n)`. This allows for the principled design of equivariant layers and nonlinearities for a much broader class of symmetries.

This repository provides the official code to reproduce the experiments in our paper, including applications to LorentzNet and velocity learning tasks.

---

## Core Concept: Invariant Bilinear Forms for `gl(n)`

A key contribution of our work is a systematic way to define a non-degenerate, adjoint-invariant bilinear form—a generalization of the Killing form—for any matrix Lie algebra. This form is crucial for building equivariant nonlinearities and invariants.

For example, for the general linear group `gl(n)`, which is reductive but not semi-simple, the standard Killing form is degenerate. We define a non-degenerate form as:

`B(X, Y) = 2n * tr(XY) - tr(X)tr(Y)`

This allows us to build powerful equivariant modules for a wider range of groups. Here’s a simple code snippet demonstrating how this is used to create an invariant feature:

```python
# From core/layers.py
import torch

class LNInvariant(nn.Module):
    """
    Computes an invariant scalar feature from a Lie algebra element
    using the 'self-killing' method with our non-degenerate form.
    """
    def __init__(self, in_channels, algebra_type='gl3'):
        super(LNInvariant, self).__init__()
        self.hat_layer = HatLayer(algebra_type)
        self.algebra_type = algebra_type

    def forward(self, x):
        """
        Input x: Lie algebra vectors of shape [B, F, N, 1]
        Output: Invariant scalars of shape [B, F]
        """
        # 1. Map vector representation to matrix representation
        x_hat = self.hat_layer(x) # [B, F, N, n, n]

        # 2. Compute the invariant using the bilinear form B(X, X)
        # killingform() function handles different algebra types
        invariant_scalar = killingform(x_hat, x_hat, self.algebra_type) # [B, F, N, 1]

        # 3. Aggregate over the spatial/node dimension (e.g., via mean or max)
        return invariant_scalar.mean(dim=[2, 3])
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

### LorentzNet Experiment
To reproduce the LorentzNet experiment from our paper, run the following command:
```bash
python experiments/lorentznet/train.py --config configs/lorentz_config.yaml
```

### Velocity Learning Experiment
To run the velocity learning task:
```bash
python experiments/velocity_learning/train.py --config configs/velocity_config.yaml
```

*Tip: We found that a lower learning rate (e.g., `1e-4` to `3e-5`) and careful weight initialization are crucial for stable training, especially for deeper networks.*

---

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{yourname2025reln,
  title={{Reductive Lie Neurons}: A Framework for General Matrix Lie Groups},
  author={Kim, Chankyo and Zhao, Sicheng and ...},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```