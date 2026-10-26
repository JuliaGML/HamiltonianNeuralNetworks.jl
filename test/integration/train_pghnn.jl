# One training run of a generalized Hamiltonian network on a `ParametricDataLoader`, which is the
# path that ties the pieces together: the batch splitter, the parametric loss, the `Zygote` pullback
# through the generated gradients of the energies, and the optimizer step over a *nested* parameter
# set. Each layer of the chain holds a whole network, and a forcing layer holds one of its own, so a
# step that reaches only the outer level of the tree leaves every leaf where it was.

using HamiltonianNeuralNetworks
using GeometricMachineLearning: Adam, Batch, NeuralNetwork, ParametricDataLoader,
                                TrainingOptimizer, params, train!
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/parameter_leaves.jl")

Random.seed!(1234)

# The largest change of any leaf, which a training step that never reaches the parameters leaves at
# zero.
function largest_change(before, after)
    change = Ref(0.0)
    foreach_parameter(before, after) do earlier, later
        change[] = max(change[], maximum(abs, later .- earlier))
    end
    change[]
end

# `n_integrators = 2`, so the chain holds four layers that each carry a network of their own: every
# leaf of every one of them moves.
@testset "train! moves every leaf of a generalized Hamiltonian network ($T)" for T in REAL_ELTYPES
    data = rand(T, 4, 5, 3)
    parameters = [(m = T(1.0),), (m = T(1.5),), (m = T(2.0),)]
    architecture = GeneralizedHamiltonianArchitecture(4;
        n_integrators = 2, parameters = (m = T(1.0),))
    nn = NeuralNetwork(architecture, T)
    before = deepcopy(params(nn))

    n_epochs = 2
    history = train!(nn, TrainingOptimizer(params(nn); algorithm = Adam()),
        ParametricDataLoader(data, parameters), Batch(2), n_epochs; show_progress = false)

    @test length(history) == n_epochs
    @test eltype(history) === T
    @test all(isfinite, history)
    @test all(>(zero(T)), history)

    # every leaf moved: a gradient that loses the nesting moves the first level and no leaf below it
    @test length(collect(keys(params(nn)))) == 4
    foreach_parameter(before, params(nn)) do earlier, later
        @test later != earlier
    end
end

@testset "train! moves the forcing layers of a forced network too ($T)" for T in REAL_ELTYPES
    data = rand(T, 4, 5, 3)
    parameters = [(m = T(1.0),), (m = T(1.5),), (m = T(2.0),)]
    architecture = ForcedGeneralizedHamiltonianArchitecture(4;
        n_integrators = 2, n_forcing_layers = 1, forcing_type = :P,
        parameters = (m = T(1.0),))
    nn = NeuralNetwork(architecture, T)
    before = deepcopy(params(nn))

    history = train!(nn, TrainingOptimizer(params(nn); algorithm = Adam()),
        ParametricDataLoader(data, parameters), Batch(2), 2; show_progress = false)

    @test all(isfinite, history)
    @test eltype(history) === T
    foreach_parameter(before, params(nn)) do earlier, later
        @test later != earlier
    end
    # both forcing layers, at their own level of the tree
    @test largest_change(deepcopy(before).L2, params(nn).L2) > zero(T)
    @test largest_change(deepcopy(before).L5, params(nn).L5) > zero(T)
end

# A `(q, p)` loader is the other shape the batch comes in, and the network takes it too.
@testset "train! on a (q, p) ParametricDataLoader ($T)" for T in REAL_ELTYPES
    data = (q = rand(T, 2, 5, 3), p = rand(T, 2, 5, 3))
    parameters = [(m = T(1.0),), (m = T(1.5),), (m = T(2.0),)]
    nn = NeuralNetwork(GeneralizedHamiltonianArchitecture(4; parameters = (m = T(1.0),)), T)
    before = deepcopy(params(nn))
    history = train!(nn, TrainingOptimizer(params(nn); algorithm = Adam()),
        ParametricDataLoader(data, parameters), Batch(2), 2; show_progress = false)
    @test all(isfinite, history)
    @test largest_change(before, params(nn)) > zero(T)
end
