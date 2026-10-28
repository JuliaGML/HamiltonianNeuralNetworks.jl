# The layerwise `SymbolicPullback` of a generalized Hamiltonian network.
#
# `SymbolicNeuralNetworks` composes the pullback of a chain layer by layer, putting fresh symbolic
# variables between two layers instead of inlining one layer's expression into the next. Its seam is
# one plain vector by default, so a layer that also threads the parameters of the system on to the
# next one has to say how it meets that seam; `src/pullbacks/layerwise_seam.jl` is that declaration,
# and this file checks it — the declarations themselves, the guard of the state-alone call, the
# gradient against `Zygote` in both precisions, and the refusals that remain.
#
# The reference is `Zygote` throughout. The symbolic pullbacks are built once, outside the loop over
# precisions: they are symbolic, so their construction does not depend on `T`, and only the numeric
# model, the data and the reference do.

using HamiltonianNeuralNetworks
using GeometricMachineLearning: NeuralNetwork, ParametricLoss, apply_parametric
using AbstractNeuralNetworks: Chain, FeedForwardLoss, params
using GeometricBase: NullParameters
using NeuralNetworkParameters: flatten
using LinearAlgebra: norm
using Test
import AbstractNeuralNetworks
import Random
import SymbolicNeuralNetworks
import SymbolicNeuralNetworks: SymbolicNeuralNetwork, symbolic_steps, composes_layerwise,
                               checked_layer_seed, layerwise_gradient_function, loss_seed
import Zygote

include("../helpers/eltypes.jl")

Random.seed!(1234)

const SYSTEM_PARAMETERS = (m = 1.0, ω = π / 2)

# The largest entry of the flat difference is compared against `1e-14 * eps(T) / eps(Float64)`: a
# relative `1e-14` in `Float64`, and the same multiple of `eps` in any other precision.
tolerance(::Type{T}) where {T} = 1e-14 * eps(T) / eps(Float64)

# The pullback is a function of the *output sensitivity*, which the layerwise construction ignores
# because the loss is scalar-valued. Asking it with the other float type is therefore free, and what
# comes back still has to have the parameters' element type.
other_precision(::Type{Float64}) = 1.0f0
other_precision(::Type{Float32}) = 1.0

flat(gradient) = flatten(gradient)[1]
maximum_difference(a, b) = maximum(abs, flat(a) - flat(b))
construction(pullback) = typeof(pullback.fun.gradient_function).name.name

const PARAMETER_FREE = GeneralizedHamiltonianArchitecture(4; width = 2, nhidden = 1)
const TWO_INTEGRATORS = GeneralizedHamiltonianArchitecture(4; width = 2, nhidden = 1,
    n_integrators = 2)
const PARAMETRIZED = GeneralizedHamiltonianArchitecture(4; width = 2, nhidden = 1,
    parameters = SYSTEM_PARAMETERS)
const PARAMETRIZED_TWO_INTEGRATORS = GeneralizedHamiltonianArchitecture(4; width = 2,
    nhidden = 1, n_integrators = 2, parameters = SYSTEM_PARAMETERS)
const FORCED_TWO_INTEGRATORS = ForcedGeneralizedHamiltonianArchitecture(
    4; width = 2, nhidden = 1,
    n_integrators = 2, parameters = SYSTEM_PARAMETERS)

const LAYERWISE = SymbolicNeuralNetworks.SymbolicPullback(
    SymbolicNeuralNetwork(PARAMETER_FREE), FeedForwardLoss(); layerwise = :auto)
const MONOLITHIC = SymbolicNeuralNetworks.SymbolicPullback(
    SymbolicNeuralNetwork(PARAMETER_FREE), FeedForwardLoss(); layerwise = false)
const TWO_INTEGRATOR_PULLBACK = SymbolicNeuralNetworks.SymbolicPullback(
    SymbolicNeuralNetwork(TWO_INTEGRATORS), FeedForwardLoss(); layerwise = true)
const PARAMETRIZED_PULLBACK = SymbolicNeuralNetworks.SymbolicPullback(
    SymbolicNeuralNetwork(PARAMETRIZED), ParametricLoss(); layerwise = true)
const PARAMETRIZED_TWO_INTEGRATOR_PULLBACK = SymbolicNeuralNetworks.SymbolicPullback(
    SymbolicNeuralNetwork(PARAMETRIZED_TWO_INTEGRATORS), ParametricLoss(); layerwise = :auto)

# Every layer of a parametrized chain can be seeded, and the chain composes: that is what the seam
# the layers declare buys. Structural, so it runs once.
@testset "a parametrized chain decomposes and every layer can be seeded: $n integrator(s)" for n in (1, 2)
    architecture = GeneralizedHamiltonianArchitecture(4; width = 2, nhidden = 1,
        n_integrators = n, parameters = SYSTEM_PARAMETERS)
    snn = SymbolicNeuralNetwork(architecture)
    sparams = params(snn)

    # two steps per integrator: one `SymplecticEulerA`, one `SymplecticEulerB`
    @test length(symbolic_steps(snn)) == 2n
    @test composes_layerwise(snn)
    for (layer, key) in symbolic_steps(snn)
        @test !isnothing(checked_layer_seed(layer, key, sparams[key]))
    end
    @test !isnothing(layerwise_gradient_function(snn, ParametricLoss()))
end

# A parameter-free network builds both constructions, and the two agree with each other and with
# `Zygote`. `:auto` is the layerwise one.
@testset "a parameter-free GHNN agrees with the monolithic path and with Zygote ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(PARAMETER_FREE, T)
    input, output = rand(T, 4), rand(T, 4)
    seed = other_precision(T)

    @test construction(LAYERWISE) == :LayerwiseGradientFunction
    @test construction(MONOLITHIC) != :LayerwiseGradientFunction

    gradient = LAYERWISE.fun(input, output, params(nn))(seed)
    monolithic_gradient = MONOLITHIC.fun(input, output, params(nn))(seed)
    @test eltype(flat(gradient)) === T
    @test eltype(flat(monolithic_gradient)) === T
    @test maximum_difference(gradient, monolithic_gradient) < tolerance(T)

    reference = Zygote.gradient(p -> FeedForwardLoss()(nn.model, p, input, output),
        params(nn))[1]
    @test maximum_difference(gradient, reference) < tolerance(T)
end

# Two integrators are four steps. This package's own constructor keeps refusing such a chain and names
# where to go instead; the layerwise construction builds it.
@testset "a chain of two integrators builds ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(TWO_INTEGRATORS, T)
    input, output = rand(T, 4), rand(T, 4)
    seed = other_precision(T)

    @test length(TWO_INTEGRATOR_PULLBACK.fun.gradient_function.steps) == 4
    # the sweep never asks the first layer for the sensitivity to its input
    @test isnothing(first(TWO_INTEGRATOR_PULLBACK.fun.gradient_function.steps).dλ)

    gradient = TWO_INTEGRATOR_PULLBACK.fun(input, output, params(nn))(seed)
    @test eltype(flat(gradient)) === T
    reference = Zygote.gradient(p -> FeedForwardLoss()(nn.model, p, input, output),
        params(nn))[1]
    @test maximum_difference(gradient, reference) < tolerance(T)

    @test_throws ArgumentError SymbolicPullback(PARAMETRIZED_TWO_INTEGRATORS, ParametricLoss())
    @test_throws r"SymbolicNeuralNetworks\.SymbolicPullback\(SymbolicNeuralNetwork\(arch\), ParametricLoss\(\)\)` composes the pullback layer by layer instead, with one `μ` per call" SymbolicPullback(
        PARAMETRIZED_TWO_INTEGRATORS, ParametricLoss())

    # A forced chain holds a `ForcingLayer` between two steps, which the layerwise construction cannot
    # seed, so the refusal names `ZygotePullback` alone.
    @test_throws ArgumentError SymbolicPullback(FORCED_TWO_INTEGRATORS, ParametricLoss())
    @test_throws r"cannot seed this architecture's `ForcingLayer`" SymbolicPullback(
        FORCED_TWO_INTEGRATORS, ParametricLoss())
    @test_throws ArgumentError SymbolicNeuralNetworks.SymbolicPullback(
        SymbolicNeuralNetwork(FORCED_TWO_INTEGRATORS), ParametricLoss(); layerwise = true)
    @test_throws r"`ForcingLayer`\) cannot be seeded" SymbolicNeuralNetworks.SymbolicPullback(
        SymbolicNeuralNetwork(FORCED_TWO_INTEGRATORS), ParametricLoss(); layerwise = true)
end

# The parameters of the system reach the gradient: they cross the seam as the carried datum, in the
# element type of the state even though `μ` is `Float64`.
@testset "a parametrized GHNN: the carried system parameters reach the gradient ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(PARAMETRIZED, T)
    input, output = rand(T, 4), rand(T, 4)
    seed = other_precision(T)

    @test construction(PARAMETRIZED_PULLBACK) == :LayerwiseGradientFunction

    gradient = PARAMETRIZED_PULLBACK.fun((input, SYSTEM_PARAMETERS), output, params(nn))(seed)
    @test eltype(flat(gradient)) === T
    reference = Zygote.gradient(
        p -> norm(apply_parametric(nn.model, input, SYSTEM_PARAMETERS, p) - output) /
             norm(output), params(nn))[1]
    @test maximum_difference(gradient, reference) < tolerance(T)

    # A matrix of three samples with one `μ`: the parameters of the system are repeated under every
    # column, and the pullback of the batch is the sum of the pullbacks of its columns.
    batch, batch_output = rand(T, 4, 3), rand(T, 4, 3)
    batch_gradient = flat(PARAMETRIZED_PULLBACK.fun(
        (batch, SYSTEM_PARAMETERS), batch_output,
        params(nn))(seed))
    column_gradient = sum(flat(PARAMETRIZED_PULLBACK.fun((batch[:, i], SYSTEM_PARAMETERS),
                              batch_output[:, i], params(nn))(seed)) for i in 1:3)
    @test eltype(batch_gradient) === T
    @test eltype(column_gradient) === T
    @test maximum(abs, batch_gradient - column_gradient) < tolerance(T)

    # The monolithic construction cannot be built for this architecture at all: it traces the chain
    # from a plain vector, so the layers default the parameters of the system away and the energy
    # network's first layer is short of the components it reads.
    @test_throws ArgumentError SymbolicNeuralNetworks.SymbolicPullback(
        SymbolicNeuralNetwork(PARAMETRIZED), FeedForwardLoss(); layerwise = false)
    @test_throws r"^ArgumentError: the energy network of this layer takes 4 inputs and this call has 2\.$" SymbolicNeuralNetworks.SymbolicPullback(
        SymbolicNeuralNetwork(PARAMETRIZED), FeedForwardLoss(); layerwise = false)
end

# `ParametricLoss` applies a loss in its five-argument form, so the layerwise construction cannot
# guess its expression and it declares one. Without the declaration the build declines, and the
# construction it falls back to does not build here.
@testset "`ParametricLoss` declares its expression, so the sweep can seed it ($T)" for T in REAL_ELTYPES
    snn = SymbolicNeuralNetwork(PARAMETRIZED_TWO_INTEGRATORS)
    nn = NeuralNetwork(PARAMETRIZED_TWO_INTEGRATORS, T)
    input, output = rand(T, 4), rand(T, 4)

    @test !isnothing(SymbolicNeuralNetworks.loss_expression(ParametricLoss(), [1.0], [1.0]))
    @test !isnothing(loss_seed(ParametricLoss(), snn))
    @test construction(PARAMETRIZED_TWO_INTEGRATOR_PULLBACK) == :LayerwiseGradientFunction

    gradient = PARAMETRIZED_TWO_INTEGRATOR_PULLBACK.fun((input, SYSTEM_PARAMETERS), output,
        params(nn))(other_precision(T))
    @test eltype(flat(gradient)) === T
    reference = Zygote.gradient(
        p -> ParametricLoss()(nn.model, p, input, output, SYSTEM_PARAMETERS), params(nn))[1]
    @test maximum_difference(gradient, reference) < tolerance(T)
end

# The declarations of `src/pullbacks/layerwise_seam.jl`, checked directly rather than through a
# gradient: what a layer carries at the seam, what it is applied to there, and what its generated
# kernels are called with.
@testset "the seam the layers declare" begin
    free_layer = first(symbolic_steps(SymbolicNeuralNetwork(PARAMETER_FREE)))[1]
    parametrized_layer = first(symbolic_steps(SymbolicNeuralNetwork(PARAMETRIZED)))[1]
    state, matrix = rand(4), rand(4, 3)

    # a system without parameters carries nothing, and says so with `()`
    @test @inferred(SymbolicNeuralNetworks.carried_variables(free_layer)) === ()
    @test free_layer.parameter_length == 0
    @test isconcretetype(fieldtype(typeof(free_layer), :parameter_layout))
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(free_layer, state)) == (state,)
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        free_layer, (state, SYSTEM_PARAMETERS))) == (state,)
    # the pair a parameter-free chain hands its later layers: the state and no parameters
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        free_layer, (state, NullParameters()))) == (state,)
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        free_layer, (matrix, NullParameters()))) == (matrix,)

    # one carried array of `parameter_length` symbolic variables
    carried = SymbolicNeuralNetworks.carried_variables(parametrized_layer)
    @test length(carried) == 1
    @test length(first(carried)) == 2
    @test parametrized_layer.parameter_length == 2
    @test isconcretetype(fieldtype(typeof(parametrized_layer), :parameter_layout))

    # the seam is written in the shape `μ` has, and called with the flat one
    flat_parameters = [1.0, π / 2]
    @test last(SymbolicNeuralNetworks.seam_value(parametrized_layer, [0.0], flat_parameters)) ==
          SYSTEM_PARAMETERS
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        parametrized_layer, (state, SYSTEM_PARAMETERS))) == (state, flat_parameters)
    # one set of parameters for a matrix of samples is repeated under each of its columns
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        parametrized_layer, (matrix, SYSTEM_PARAMETERS))) ==
          (matrix, repeat(flat_parameters, 1, 3))

    # one parameter is one carried value, and the seam is two arguments long all the same
    one_parameter = (m = 1.0,)
    one_parameter_architecture = GeneralizedHamiltonianArchitecture(
        4; width = 2, nhidden = 1,
        parameters = one_parameter)
    one_parameter_layer = first(symbolic_steps(SymbolicNeuralNetwork(
        one_parameter_architecture)))[1]
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        one_parameter_layer, (state, one_parameter))) == (state, [1.0])
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        one_parameter_layer, (matrix, one_parameter))) == (matrix, repeat([1.0], 1, 3))

    # the generated kernels take the state and then exactly what `carried_variables` declared
    seams = ((free_layer, state), (free_layer, (state, NullParameters())),
        (free_layer, (state, SYSTEM_PARAMETERS)),
        (parametrized_layer, (state, SYSTEM_PARAMETERS)),
        (parametrized_layer, (matrix, SYSTEM_PARAMETERS)))
    for (layer, x) in seams
        @test length(SymbolicNeuralNetworks.seam_arguments(layer, x)) ==
              1 + length(SymbolicNeuralNetworks.carried_variables(layer))
    end

    # parameters that flatten to no entries are no parameters, and carry nothing
    empty_parameters = (a = Float64[],)
    empty_layer = first(Chain(GeneralizedHamiltonianArchitecture(4; width = 2, nhidden = 1,
        parameters = empty_parameters)).layers)
    @test @inferred(SymbolicNeuralNetworks.carried_variables(empty_layer)) === ()
    @test @inferred(SymbolicNeuralNetworks.seam_arguments(
        empty_layer, (state, empty_parameters))) == (state,)

    # the state alone is a seam only for a system without parameters
    @test_throws ArgumentError SymbolicNeuralNetworks.seam_arguments(parametrized_layer, state)
    @test_throws r"^ArgumentError: the system of this `SymplecticEuler` has 2 parameter\(s\), so the " *
                 r"pullback has to be given `\(state, system parameters\)` rather than the state alone\.$" SymbolicNeuralNetworks.seam_arguments(
        parametrized_layer, state)

    # `_seam_state` stacks a `(q, p)` pair whatever the pair of array types is, `p` below `q`
    stacked = rand(4, 3)
    pair = (q = @view(stacked[1:2, :]), p = rand(2, 3))
    @test HamiltonianNeuralNetworks._seam_state(pair) == vcat(pair.q, pair.p)
    @test HamiltonianNeuralNetworks._seam_state(state) === state

    # the energy a step is built from carries the same pair of fields
    energy = HamiltonianNeuralNetworks.SymbolicKineticEnergy(4, 2, 1,
        AbstractNeuralNetworks.TanhActivation(); parameters = SYSTEM_PARAMETERS)
    @test isconcretetype(fieldtype(typeof(energy), :parameter_layout))
    @test energy.parameter_length == 2
end
