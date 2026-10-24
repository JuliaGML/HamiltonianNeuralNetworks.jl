using HamiltonianNeuralNetworks
using HamiltonianNeuralNetworks: HNNLoss, SymplecticEulerLoss
using GeometricMachineLearning: CPU
using AbstractNeuralNetworks: NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

# A loss's `ps` is a `NetworkParameters`, so a bare `NamedTuple` reaches no method of a loss of
# this package rather than a silently different gradient: which of the two AD hands the loss is
# what an untyped argument hides. The four-argument methods of a training loss carry the same
# annotation, and their model slot is reached by `AbstractNeuralNetworks`' own untyped fallback
# where the wrapper is missing.
#
# Not numeric: the element type of the data does not reach the method selection, so this is not run
# over `REAL_ELTYPES`.
@testset "a loss takes a NetworkParameters" begin
    Random.seed!(23)
    hnn_arch = StandardHamiltonianArchitecture(2, 4, 1)

    hnn_input, hnn_output = randn(2, 3), randn(2, 3)

    losses = (
        (HNNLoss(hnn_arch), hnn_arch, hnn_input, hnn_output),
        (SymplecticEulerLoss(hnn_arch, 0.1), hnn_arch, hnn_input, hnn_output)
    )
    for (loss, arch, input, output) in losses
        nn = NeuralNetwork(arch, CPU(), Float64)
        ps = params(nn)
        # the wrapper is accepted, and the same number comes back through the `NeuralNetwork`
        # method, which forwards to the wrapper
        @test loss(nn, input, output) == loss(ps, input, output)
        # the bare `NamedTuple` under the wrapper is not
        @test_throws MethodError loss(params(ps), input, output)
    end
end
