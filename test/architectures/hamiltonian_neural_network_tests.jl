using HamiltonianNeuralNetworks
using GeometricMachineLearning: DataLoader, NetworkLoss, NetworkParameters, NeuralNetwork
import AbstractNeuralNetworks
using Test
using Zygote
import Random
Random.seed!(1234)

include("../helpers/eltypes.jl")

const GMLA = AbstractNeuralNetworks.Activation
function allocate_network_and_data_loader(
        T::Type, dim::Integer, width::Integer, nhidden::Integer, activation::GMLA)
    arch = StandardHamiltonianArchitecture(dim, width, nhidden, activation)
    loss = HNNLoss(arch)
    input = rand(T, dim, 10)
    output = rand(T, dim, 10)
    dl = DataLoader(input, output)
    NeuralNetwork(arch, T), loss, dl
end

"""
This tests if we can call the HNN loss.
"""
function test_hnn_loss(T::Type, dim::Integer = 2,
        width::Integer = dim,
        nhidden::Integer = 1,
        activation::GMLA = AbstractNeuralNetworks.SigmoidActivation())
    nn, loss, dl = allocate_network_and_data_loader(T, dim, width, nhidden, activation)
    @test typeof(loss(nn.params, dl.input, dl.output)) <: Real
    @test loss(nn.params, dl.input, dl.output) isa T
    @test typeof(loss) <: NetworkLoss
end

"""
This tests if we can differentiate the HNN loss.
"""
function test_hnn_loss_derivative(T::Type, dim::Integer = 2,
        width::Integer = dim,
        nhidden::Integer = 1,
        activation::GMLA = AbstractNeuralNetworks.SigmoidActivation())
    nn, loss, dl = allocate_network_and_data_loader(T, dim, width, nhidden, activation)
    dp = Zygote.gradient(ps -> loss(ps, dl.input, dl.output), nn.params)[1]
    @test typeof(dp) <: NetworkParameters
    @test keys(dp) == keys(nn.params)
    @test all(layer -> all(leaf -> eltype(leaf) === T, values(dp[layer])), keys(dp))
end

"""
This tests if we can build the symbolic pullback of the HNN loss and evaluate it.
"""
function test_hnn_symbolic_pullback(T::Type, dim::Integer = 2,
        width::Integer = dim,
        nhidden::Integer = 1,
        activation::GMLA = AbstractNeuralNetworks.SigmoidActivation())
    arch = StandardHamiltonianArchitecture(dim, width, nhidden, activation)
    nn = NeuralNetwork(arch, T)
    pb = SymbolicPullback(arch)
    input = rand(T, dim, 10)
    output = rand(T, dim, 10)
    loss_value, back = pb(nn.params, nn.model, (input, output))
    @test typeof(loss_value) <: Real
    @test loss_value isa T
    @test keys(back(1)) == keys(nn.params)
    @test all(layer -> all(leaf -> eltype(leaf) === T, values(back(1)[layer])), keys(nn.params))
end

@testset "HNNLoss, its gradient and its symbolic pullback ($T)" for T in REAL_ELTYPES
    test_hnn_loss(T)
    test_hnn_loss_derivative(T)
    test_hnn_symbolic_pullback(T)
end
