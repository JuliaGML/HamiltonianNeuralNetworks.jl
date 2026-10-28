"""
    HamiltonianNeuralNetworks

Hamiltonian neural networks: an architecture whose network is a Hamiltonian and whose vector field
is its symplectic gradient, the generalized and forced Hamiltonian networks whose forward pass also
takes the parameters of the system, the losses that train them, and their symbolic pullbacks.
"""
module HamiltonianNeuralNetworks

import AbstractNeuralNetworks
import AbstractNeuralNetworks: Chain, initialparameters, parameterlength
using AbstractNeuralNetworks: AbstractExplicitLayer, Activation, Architecture,
                              ArrayOrNamedTuple, Dense, Linear, NamedTupleOfArrays,
                              NetworkLoss, NeuralNetworkBackend, NeuralNetwork,
                              input_dimension,
                              output_dimension, params
using GeometricBase: NullParameters, OptionalParameters
import GeometricMachineLearning
using GeometricMachineLearning: ForcingLayer, ParametricLoss, apply_parametric
using LinearAlgebra: I, norm, pinv
using NeuralNetworkParameters: NetworkParameters, mapstorage, parameter_eltype, unflatten
import Random
import SymbolicNeuralNetworks
import SymbolicNeuralNetworks: SymbolicPullback
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, derivative
import Symbolics

export HamiltonianArchitecture, StandardHamiltonianArchitecture,
       GeneralizedHamiltonianArchitecture, ForcedGeneralizedHamiltonianArchitecture
export HNNLoss, SymplecticEulerLoss
export SymbolicPullback

include("architectures/hamiltonian_neural_network.jl")
include("architectures/generalized_hamiltonian_neural_network.jl")
include("architectures/forced_generalized_hamiltonian_neural_network.jl")
include("loss/hnn_loss.jl")
include("loss/symplectic_euler_loss.jl")
include("pullbacks/symbolic_hnn_pullback.jl")
include("pullbacks/layerwise_seam.jl")

end
