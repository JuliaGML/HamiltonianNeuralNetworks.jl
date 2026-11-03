# HamiltonianNeuralNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGML/HamiltonianNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGML/HamiltonianNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

Hamiltonian neural networks learn a Hamiltonian vector field from data. The network approximates
the Hamiltonian, and the vector field is its symplectic gradient, so the learned dynamics are
Hamiltonian by construction.

The package provides

- the architectures `StandardHamiltonianArchitecture`, `GeneralizedHamiltonianArchitecture` and
  `ForcedGeneralizedHamiltonianArchitecture`,
- the losses `HNNLoss`, for vector field data, and `SymplecticEulerLoss`, for pairs of states one
  timestep apart,
- the `SymbolicPullback`s that compute the gradient of `HNNLoss` symbolically and of
  `ParametricLoss` in closed form.

The networks train with the data loader, the training loop and the optimisers of
[GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl).

## Installation

The examples call GeometricMachineLearning directly, so add it beside the package:

```julia
using Pkg
Pkg.add(["HamiltonianNeuralNetworks", "GeometricMachineLearning"])
```

## Example

Learn the harmonic oscillator, whose vector field at a state `z = (q, p)` is `𝕁 * z`, from the
vector field at 100 random states:

```julia
using HamiltonianNeuralNetworks
using GeometricMachineLearning: Adam, Batch, DataLoader, NeuralNetwork, TrainingOptimizer, train!

𝕁 = [0.0 1.0; -1.0 0.0]              # the canonical Poisson tensor of ℝ²
input = randn(2, 100)                 # 100 states, one per column
output = 𝕁 * input                    # the vector field at each state

arch = StandardHamiltonianArchitecture(2, 5)
nn = NeuralNetwork(arch)
dl = DataLoader(input, output)
loss_history = train!(nn, TrainingOptimizer(nn.params; algorithm = Adam()), dl, Batch(10), 100;
    loss = HNNLoss(arch))
```

`loss_history` holds the loss of each epoch.

## Documentation

The [tutorial](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest/tutorials/hamiltonian_neural_network/)
trains on vector field data and on phase space data. The
[manual](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest) describes the
architectures and the losses.
