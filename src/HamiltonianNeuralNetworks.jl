"""
    HamiltonianNeuralNetworks

Hamiltonian neural networks: an architecture whose network is a Hamiltonian and whose vector field
is its symplectic gradient, the losses that train it, and its symbolic pullback.
"""
module HamiltonianNeuralNetworks

import AbstractNeuralNetworks
import AbstractNeuralNetworks: Chain
using AbstractNeuralNetworks: AbstractExplicitLayer, Activation, Architecture,
                              ArrayOrNamedTuple, Dense, Linear, NetworkLoss, input_dimension
using LinearAlgebra: I, norm, pinv
using NeuralNetworkParameters: NetworkParameters, mapstorage, parameter_eltype
import SymbolicNeuralNetworks
import SymbolicNeuralNetworks: SymbolicPullback
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, derivative
import Symbolics

export HamiltonianArchitecture, StandardHamiltonianArchitecture,
       GeneralizedHamiltonianArchitecture
export HNNLoss, SymplecticEulerLoss
export SymbolicPullback

include("architectures/hamiltonian_neural_network.jl")
include("loss/hnn_loss.jl")
include("loss/symplectic_euler_loss.jl")
include("pullbacks/symbolic_hnn_pullback.jl")

end
