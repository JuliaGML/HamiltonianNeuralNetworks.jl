# The layerwise pullback of a generalized Hamiltonian network: the seam its layers declare to the
# sweep of `SymbolicNeuralNetworks`, and the pullback of `ParametricLoss` that the sweep composes for
# a chain of more than one integrator — its value and gradient against `Zygote` on both batch forms,
# and `train!` with it against `train!` with `ZygotePullback`.
#
# The symbolic pullbacks are built once, outside the loop over precisions: they are symbolic, so
# their construction does not depend on `T`, and only the numeric model, the data and the reference
# do.

using HamiltonianNeuralNetworks
using GeometricMachineLearning: Batch, NeuralNetwork, ParametricDataLoader, ParametricLoss,
                                TrainingOptimizer, ZygotePullback, train!
using GeometricMachineLearning: Adam
using AbstractNeuralNetworks: Chain, params
using GeometricBase: NullParameters
using NeuralNetworkParameters: flatten
using Test
import Random
import SymbolicNeuralNetworks
using SymbolicNeuralNetworks: SymbolicNeuralNetwork

include("../helpers/eltypes.jl")

Random.seed!(1234)

const SYSTEM_PARAMETERS = (m = 1.0, ω = π / 2)
# three samples with distinct values of `μ`: a gradient that paired a sample with another sample's
# `μ`, or used one `μ` for the whole batch, is a different gradient
const SAMPLE_PARAMETERS = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0), (m = 3.0, ω = 1.0)]

# The largest entry of the flat difference is compared against `1e-14 * eps(T) / eps(Float64)`: an
# absolute `1e-14` in `Float64`, and the same multiple of `eps` in any other precision.
tolerance(::Type{T}) where {T} = 1e-14 * eps(T) / eps(Float64)

# The pullback returns a function of the sensitivity of the loss. A seed of the other float type
# still gives a gradient of the parameters' element type.
other_precision(::Type{Float64}) = 1.0f0
other_precision(::Type{Float32}) = 1.0

flat(gradient) = flatten(gradient)[1]
maximum_difference(a, b) = maximum(abs, flat(a) - flat(b))

function architecture(n_integrators; parameters = SYSTEM_PARAMETERS)
    GeneralizedHamiltonianArchitecture(4; width = 3, nhidden = 1, n_integrators, parameters)
end

# Two integrators are four steps, and every seam a longer chain has: each kind of step passes the
# parameters of the system on to the next, and the last step returns the state alone.
const TWO_INTEGRATORS = SymbolicPullback(architecture(2), ParametricLoss())

# The pullback of a generalized architecture is the sweep, whatever its number of integrators: two
# steps per integrator, and no sensitivity of the first step's input, which a parameter gradient has
# no use for.
@testset "the pullback of a chain of $n integrator(s) is the layerwise sweep" for n in (1, 2, 3)
    pullback = n == 2 ? TWO_INTEGRATORS :
               SymbolicPullback(architecture(n), ParametricLoss())
    @test pullback.fun isa HamiltonianNeuralNetworks._LayerwiseParameterGradient
    sweep = pullback.fun.gradient_function
    @test sweep isa SymbolicNeuralNetworks.LayerwiseGradientFunction
    @test length(sweep.steps) == 2n
    @test isnothing(first(sweep.steps).dλ)
end

# The reference is `ZygotePullback(ParametricLoss())` on the same batch, a stacked one and a `(q, p)`
# one, whose loss weighs the two halves by their own residual norms.
@testset "a chain of two integrators against Zygote ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(architecture(2), T)
    pullback = TWO_INTEGRATORS
    input, output = rand(T, 4, 1, 3), rand(T, 4, 1, 3)
    pair(x) = (q = x[1:2, :, :], p = x[3:4, :, :])
    for (x, y) in ((input, output), (pair(input), pair(output)))
        value, back = pullback(params(nn), nn.model, (x, y, SAMPLE_PARAMETERS))
        reference_value, reference_back = ZygotePullback(ParametricLoss())(
            params(nn), nn.model, (x, y, SAMPLE_PARAMETERS))
        @test value isa T
        @test value ≈ reference_value rtol = 8 * eps(T)

        gradient = back(one(value))
        reference = reference_back(one(reference_value))
        @test eltype(flat(gradient)) === T
        @test maximum_difference(gradient, reference) < tolerance(T)
        # the comparison above would also pass for a zero gradient against a zero reference
        @test maximum(abs, flat(reference)) > sqrt(eps(T))
        @test eltype(flat(back(other_precision(T)))) === T
    end
end

# A stacked batch is a `2d × 1 × k` tensor, and a `(q, p)` batch a pair of `d × 1 × k` ones. A matrix
# of samples, or a tensor whose middle size is not one, passes the call signature in either form, and
# the pullback refuses it by naming the forms it takes and the form it got.
@testset "a matrix batch or a middle size other than one is an ArgumentError" begin
    nn = NeuralNetwork(architecture(2), Float64)
    tensor, matrix, wide = rand(4, 1, 3), rand(4, 3), rand(4, 2, 3)
    matrix_pair = (q = rand(2, 3), p = rand(2, 3))
    wide_pair = (q = rand(2, 2, 3), p = rand(2, 2, 3))
    half_wide_pair = (q = rand(2, 1, 3), p = rand(2, 2, 3))
    for (x, y) in ((matrix, matrix), (tensor, matrix), (matrix, tensor),
        (matrix_pair, matrix_pair), (wide, wide), (tensor, wide), (wide_pair, wide_pair),
        (half_wide_pair, half_wide_pair))
        batch = (x, y, SAMPLE_PARAMETERS)
        @test_throws ArgumentError TWO_INTEGRATORS(params(nn), nn.model, batch)
        message = try
            TWO_INTEGRATORS(params(nn), nn.model, batch)
        catch error
            sprint(showerror, error)
        end
        @test occursin("2d × 1 × k", message)
        @test occursin("(q, p) pair of d × 1 × k", message)
        @test occursin(summary(x), message)
        @test occursin(summary(y), message)
    end
end

# `train!` takes the pullback as its `loss`, and under one seed it trains as `ZygotePullback` does:
# the same minibatches, the same gradients to rounding, so the same history and the same parameters.
@testset "train! with the layerwise pullback ($T)" for T in REAL_ELTYPES
    n_epochs = 3
    data = rand(T, 4, 5, 3)
    # a loader takes its system parameters in the element type of its data
    loader = ParametricDataLoader(data, [map(T, μ) for μ in SAMPLE_PARAMETERS])
    function trained(loss)
        Random.seed!(42)
        nn = NeuralNetwork(architecture(2), T)
        start = copy(flat(params(nn)))
        history = train!(nn, TrainingOptimizer(params(nn); algorithm = Adam()), loader,
            Batch(4), n_epochs; loss, show_progress = false)
        history, start, flat(params(nn))
    end
    history, start, final = trained(TWO_INTEGRATORS)
    reference_history, reference_start, reference_final = trained(
        ZygotePullback(ParametricLoss()))

    @test start == reference_start
    @test history isa Vector{T}
    @test length(history) == n_epochs
    @test all(isfinite, history)
    @test history ≈ reference_history rtol = 100 * eps(T)
    @test final ≈ reference_final rtol = 100 * eps(T)
    @test final != start
end

# The declarations of `src/pullbacks/layerwise_seam.jl`, checked directly rather than through a
# gradient: what a layer carries at the seam, what it is applied to there, and what its generated
# kernels are called with.
@testset "the seam the layers declare" begin
    free_layer = first(Chain(architecture(1; parameters = NullParameters())).layers)
    layer = first(Chain(architecture(1)).layers)
    states = rand(Float32, 4, 3)

    # a system without parameters carries nothing, and says so with `()`
    @test @inferred(SymbolicNeuralNetworks.carried_variables(free_layer)) === ()
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        free_layer, (states, fill(NullParameters(), 3)))) == (states,)

    # one carried array of `parameter_length` symbolic variables, which the seam puts back into the
    # shape of `μ`
    carried = SymbolicNeuralNetworks.carried_variables(layer)
    @test length(carried) == 1
    @test length(only(carried)) == 2
    @test isconcretetype(fieldtype(typeof(layer), :parameter_layout))
    @test last(SymbolicNeuralNetworks.seam_value(layer, [0.0], [1.0, π / 2])) ==
          SYSTEM_PARAMETERS

    # the kernels are called with the state and the flat parameters of each sample in its column,
    # in the element type of the state
    arguments = @inferred SymbolicNeuralNetworks.seam_arguments(
        layer, (states, SAMPLE_PARAMETERS))
    @test length(arguments) == 1 + length(carried)
    @test first(arguments) === states
    @test last(arguments) == Float32[1 2 3; 2 3 1]
    @test eltype(last(arguments)) === Float32

    # parameters that flatten to no entries are no parameters, and carry nothing
    empty_parameters = (a = Float64[],)
    empty_layer = first(Chain(architecture(1; parameters = empty_parameters)).layers)
    @test @inferred(SymbolicNeuralNetworks.carried_variables(empty_layer)) === ()
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        empty_layer, (states, fill(empty_parameters, 3)))) == (states,)
end
