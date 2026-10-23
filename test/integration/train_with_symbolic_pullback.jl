using HamiltonianNeuralNetworks
using GeometricMachineLearning: Batch, CPU, DataLoader, GradientMethod, NetworkLoss,
                                NeuralNetwork, TrainingOptimizer, params, train!
import AbstractNeuralNetworks
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/leaves.jl")

# `train!` steps with the gradient of the `loss` it is given. One epoch of one minibatch with
# `GradientMethod` moves the parameters by `-η` times the symbolic gradient at `p₀`, and the history
# is the pullback's loss value. `test/pullbacks/symbolic_hnn_pullback.jl` tests the gradient itself
# against `ZygotePullback`'s (GeometricMachineLearning.jl issue #343).
@testset "train! trains a Hamiltonian network with SymbolicPullback ($T)" for T in REAL_ELTYPES
    Random.seed!(9)
    arch = StandardHamiltonianArchitecture(2, 4, 1, AbstractNeuralNetworks.SigmoidActivation())
    dl = DataLoader(randn(T, 2, 8), randn(T, 2, 8); suppress_info = true)
    nn = NeuralNetwork(arch, CPU(), T)
    p₀ = deepcopy(params(nn))
    pullback = SymbolicPullback(arch)
    batch = Batch(8)
    Random.seed!(2)
    indices = only(batch(dl))
    data = map(data -> cat((data[:, t:t, p] for (t, p) in indices)...; dims = 3),
        (dl.input, dl.output))
    value, back = pullback(p₀, nn.model, data)
    g = back(one(value))
    η = T(0.1)
    opt = TrainingOptimizer(params(nn); algorithm = GradientMethod(), linesearch = η)
    Random.seed!(2)
    history = train!(nn, opt, dl, batch, 1; loss = pullback, show_progress = false)
    @test history == [value]
    @test value isa T
    @test eltype(history) === T
    foreach_leaf(params(nn), p₀, g) do layer, layer₀, gradient
        @test layer ≈ layer₀ .- η .* gradient
        @test layer != layer₀
    end
end

# Without `loss`, `train!` trains on the loss that matches the architecture and the batch, and on
# any other combination it says that it needs one.
#
# Not numeric: the elements selected are loss types, not values, so this is not run over
# `REAL_ELTYPES`.
@testset "train!'s default losses" begin
    # a Hamiltonian network reaches the fallback
    for (arch, batch) in ((StandardHamiltonianArchitecture(2, 4, 1), Batch(4)),)
        @test_throws ArgumentError NetworkLoss(arch, batch)
    end
end
