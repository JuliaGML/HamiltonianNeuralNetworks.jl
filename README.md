# HamiltonianNeuralNetworks.jl

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGML/HamiltonianNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGML/HamiltonianNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

Hamiltonian neural networks learn a Hamiltonian vector field from data. The network approximates
the Hamiltonian, and the vector field is its symplectic gradient, so the learned dynamics are
Hamiltonian by construction.

The package provides the architecture `StandardHamiltonianArchitecture`, the losses `HNNLoss` (for
vector field data) and `SymplecticEulerLoss` (for pairs of states one timestep apart), and a
`SymbolicPullback` that computes the gradient of `HNNLoss` symbolically. The networks train with
the data loader, the training loop and the optimisers of
[GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl):

```julia
using HamiltonianNeuralNetworks
using GeometricMachineLearning: Adam, Batch, DataLoader, NeuralNetwork, TrainingOptimizer, train!

arch = StandardHamiltonianArchitecture(2, 5)
nn = NeuralNetwork(arch)
dl = DataLoader(input, output)    # states and the vector field at each, two 2 × N matrices
loss_history = train!(nn, TrainingOptimizer(nn.params; algorithm = Adam()), dl, Batch(10), 100;
    loss = HNNLoss(arch))
```

The [tutorial](https://juliagml.github.io/HamiltonianNeuralNetworks.jl/latest/tutorials/hamiltonian_neural_network/)
trains on vector field data and on phase space data. The package comes from
GeometricMachineLearning.jl, with the history of every file it took from there.
