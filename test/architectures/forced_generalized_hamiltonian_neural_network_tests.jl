# The forced generalized Hamiltonian network: its chain, its forward pass with and without system
# parameters for each forcing type, and what its constructor refuses.

using HamiltonianNeuralNetworks
using GeometricMachineLearning: ForcingLayer, NeuralNetwork, apply_parametric
using AbstractNeuralNetworks: params
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

const FORCED_DIM = 4
const SYSTEM_PARAMETERS = (m = 1.0,)
const FORCING_TYPES = (:Q, :P, :QP)

finite(x::AbstractArray) = all(isfinite, x)
finite(z::NamedTuple) = finite(z.q) && finite(z.p)

# The value of the `ArgumentError` that `f` raises, so that a test can require that the message names
# the bad value and not only the type.
function argument_error_message(f)
    try
        f()
    catch e
        e isa ArgumentError || rethrow()
        return e.msg
    end
    error("no `ArgumentError` was raised")
end

# The chain of one integrator is the step that changes `q`, the forcing, and the step that changes
# `p`, and the last of them returns the state alone.
@testset "the chain of a forced architecture ($forcing_type)" for forcing_type in FORCING_TYPES
    architecture = ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
        parameters = SYSTEM_PARAMETERS, forcing_type)
    chain = NeuralNetwork(architecture).model
    @test length(chain) == 3
    @test chain.layers[1] isa HamiltonianNeuralNetworks.SymplecticEuler
    @test chain.layers[2] isa ForcingLayer
    @test chain.layers[3] isa HamiltonianNeuralNetworks.SymplecticEuler

    architecture2 = ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
        parameters = SYSTEM_PARAMETERS, forcing_type, n_integrators = 2)
    chain2 = NeuralNetwork(architecture2).model
    @test length(chain2) == 6
    @test all(layer -> layer isa ForcingLayer, chain2.layers[[2, 5]])
    @test all(layer -> layer isa HamiltonianNeuralNetworks.SymplecticEuler,
        chain2.layers[[1, 3, 4, 6]])
end

# The forward pass, per forcing type and per precision: the shape of a state, its element type, a
# finite result, and — for the last step of the chain — the state alone rather than the state with
# the parameters of the system beside it. The same state with two values of `μ` gives two answers,
# which is what makes the network depend on the parameters of the system at all.
@testset "ForcedGeneralizedHamiltonianArchitecture $forcing_type ($T)" for forcing_type in FORCING_TYPES,
    T in REAL_ELTYPES

    architecture = ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
        parameters = SYSTEM_PARAMETERS, forcing_type)
    nn = NeuralNetwork(architecture, T)
    x = rand(T, FORCED_DIM)
    output = nn(x, SYSTEM_PARAMETERS)
    @test size(output) == (FORCED_DIM,)
    @test eltype(output) === T
    @test finite(output)
    @test !isapprox(nn(x, (m = 2.0,)), output)
end

# The refusal of the forcing type with the phrase of the check that raises it, the refusal of an
# odd dimension, which a forced chain has no split of the state for either, and the refusals of a
# negative number of forcing layers and of a width of zero at construction rather than in `Chain`.
@testset "the forced architecture is refused what it does not know" begin
    @test_throws ArgumentError ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
        forcing_type = :X)
    message = argument_error_message(() -> ForcedGeneralizedHamiltonianArchitecture(
        FORCED_DIM;
        forcing_type = :X))
    @test occursin(":X", message)
    @test occursin(":QP", message)

    @test_throws ArgumentError ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM + 1)
    @test occursin("$(FORCED_DIM + 1)",
        argument_error_message(
            () -> ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM + 1)))

    @test_throws ArgumentError ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
        n_forcing_layers = -1)
    message = argument_error_message(() -> ForcedGeneralizedHamiltonianArchitecture(
        FORCED_DIM; n_forcing_layers = -1))
    @test occursin("n_forcing_layers", message)
    @test occursin("-1", message)
    no_hidden_layer = ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM; n_forcing_layers = 0)
    @test no_hidden_layer.n_forcing_layers == 0

    @test_throws ArgumentError ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM; width = 0)
    @test occursin("`width` is 0",
        argument_error_message(
            () -> ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM; width = 0)))
end

# `nn(x, μ)` is `apply_parametric`, for every forcing type and every batch form a
# `ParametricDataLoader` draws. The forced architecture is a sibling of the generalized one, so
# without a functor of its own it falls through to the generic two-argument one, which reads the
# parameters of the system as the parameters of the network.
@testset "the forced functor agrees with `apply_parametric` ($T)" for T in REAL_ELTYPES
    μs = [(m = 1.0,), (m = 2.0,), (m = 3.0,)]
    for forcing_type in FORCING_TYPES
        nn = NeuralNetwork(
            ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
                parameters = SYSTEM_PARAMETERS, forcing_type),
            T)
        half = FORCED_DIM ÷ 2
        x = rand(T, FORCED_DIM)
        μ = first(μs)
        @test nn(x, μ) ≈ apply_parametric(nn.model, x, μ, params(nn))
        @test eltype(nn(x, μ)) === T
        @test size(nn(x, μ)) == size(x)

        # a `(q, p)` pair
        z = (q = rand(T, half), p = rand(T, half))
        forced = nn(z, μ)
        reference = apply_parametric(nn.model, z, μ, params(nn))
        @test keys(forced) == (:q, :p)
        @test forced.q ≈ reference.q
        @test forced.p ≈ reference.p

        # a `2d × 1 × k` tensor with three distinct values of μ
        batch = rand(T, FORCED_DIM, 1, 3)
        forced_batch = nn(batch, μs)
        @test size(forced_batch) == size(batch)
        @test forced_batch ≈ apply_parametric(nn.model, batch, μs, params(nn))
        @test !(forced_batch[:, :, 1] ≈ forced_batch[:, :, 2])
        @test all(forced_batch[:, :, i] ≈ nn(batch[:, :, i:i], [μs[i]])[:, :, 1]
        for i in axes(batch, 3))

        # `apply_parametric` takes any `AbstractVector` of μ, and `Vector{Any}` of `NamedTuple`s is
        # not a `Vector{<:NamedTuple}`.
        untyped = Any[(m = 1.0,), (m = 2.0,), (m = 3.0,)]
        @test nn(batch, untyped) ≈ apply_parametric(nn.model, batch, untyped, params(nn))
    end
end

# The chain of a forced architecture holds its forcing layers with the parameters of the network
# nested one level deeper than a step's, so a network whose leaves are not reached cannot train.
@testset "the parameters of a forcing layer are its network's ($T)" for T in REAL_ELTYPES
    architecture = ForcedGeneralizedHamiltonianArchitecture(FORCED_DIM;
        parameters = SYSTEM_PARAMETERS, forcing_type = :P, n_forcing_layers = 2,
        n_integrators = 2)
    nn = NeuralNetwork(architecture, T)
    forcing = params(nn).L2
    # a first dense layer, `n_forcing_layers` hidden ones, and the linear map back to `dim ÷ 2` rows
    @test length(forcing) == 4
    @test all(leaf -> eltype(leaf) === T, values(forcing.L1))
end
