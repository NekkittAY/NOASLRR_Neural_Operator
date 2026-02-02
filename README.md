# ResDeepONet-LowRank-Chebyshev

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A residual Deep Operator Network (DeepONet) with **Chebyshev spectral features**, **low-rank linear projections**, and **adaptive gating**, designed for efficient learning of nonlinear operators arising in partial differential equations (PDEs).

---

## Overview

**ResDeepONet-LowRank-Chebyshev** is a neural operator architecture that extends the classical DeepONet framework by enriching the branch network with:

1. **Spectral information** via Chebyshev polynomial coefficients  
2. **Low-rank projections** to reduce parameter count and improve generalization  
3. **Residual-style gated fusion** of multiple functional representations  

The model learns a nonlinear operator  
```math
G_\theta: \mathcal{A} \rightarrow \mathcal{U},
```
mapping input functions (e.g. initial conditions, coefficients) to solution fields evaluated at arbitrary spatial coordinates.

This hybrid design balances expressivity and efficiency, making it suitable for operator learning in PDE problems with limited data or high-dimensional inputs.

---

## Architecture

The network consists of three main components:

### 1. Branch Network (Function Encoder)

The branch processes the input function $\( f \in \mathbb{R}^{N} \)$ using three complementary pathways:

#### • MLP Branch  
A standard multilayer perceptron that learns nonlinear representations directly from discretized function values.

#### • Chebyshev Spectral Branch  
Computes the first $\(K\)$ Chebyshev coefficients:
```math
c_k = 2 \, \mathbb{E}[f(x) T_k(x)], \quad k=0,\dots,K-1,
```
where $\(T_k\)$ are Chebyshev polynomials of the first kind.  
This branch captures **global and low-frequency structure** of the input function.

#### • Low-Rank Branch  
A factorized linear mapping
```math
W \approx BA,\quad \text{rank}(W)=r \ll \min(d_{\text{in}}, d_{\text{out}}),
```
which provides a compact representation and reduces overfitting.

---

### 2. Adaptive Gating Mechanism

Two sigmoid-based gates are used:

- **Auxiliary gate**: blends Chebyshev and low-rank features  
- **Final gate**: fuses auxiliary features with the MLP branch  

This allows the model to **adaptively select** the most informative representation for each input function.

---

### 3. Trunk Network (Coordinate Encoder)

The trunk network encodes spatial (or spatiotemporal) coordinates  
```math
x \in \mathbb{R}^{d}
```
into a latent representation shared across all inputs.

---

### 4. Operator Evaluation

The output is computed via an inner product:
```math
u(x) = \langle h(f), t(x) \rangle,
```
where:
- $\(h(f)\)$ is the fused branch embedding
- $\(t(x)\)$ is the trunk embedding

This formulation enables evaluation at **arbitrary query points**.

---

## Algorithm

1. Encode the input function using:
   - MLP features
   - Chebyshev spectral coefficients
   - Low-rank linear projection
2. Fuse auxiliary features with adaptive gating
3. Combine with MLP branch through a residual-style gate
4. Encode spatial coordinates with the trunk network
5. Compute the output via inner product in latent space

---

## Features

- Spectral feature extraction via Chebyshev polynomials
- Parameter-efficient low-rank linear layers
- Residual DeepONet-style operator learning
- Adaptive gating for feature fusion
- Supports arbitrary evaluation grids
- Suitable for PDE surrogate modeling and operator regression

---

## Technology Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch
- **Numerical Computing**: NumPy
- **Model Type**: Neural Operators / DeepONet

---

## Applications

- Parametric PDEs (elliptic, parabolic, nonlinear)
- Operator regression in scientific machine learning
- Surrogate modeling for expensive solvers
- Learning mappings between function spaces
