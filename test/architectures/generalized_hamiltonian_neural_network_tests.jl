# The generalized Hamiltonian network: its forward pass with and without system parameters, its
# parameter counts against the thesis, what its constructor refuses, the shapes of a batch, the
# functor, and the two steps of one integrator against the gradient of their own energies.

using HamiltonianNeuralNetworks
using HamiltonianNeuralNetworks: SymbolicKineticEnergy, SymbolicPotentialEnergy,
                                 SymplecticEulerA,
                                 SymplecticEulerB
using GeometricMachineLearning: Chain, NeuralNetwork, apply_parametric, ParametricLoss
using GeometricBase: NullParameters
using AbstractNeuralNetworks: OneInitializer, parameterlength, params
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

# The output of a generalized Hamiltonian network whose weights are all one. The two steps of its
# single integrator enter it, each through the generated gradient of its own energy, so a change to
# the order of the steps, to the sign of a gradient, to the shape of an energy network or to the
# entries of the input its gradient is taken over moves it.
const GHNN_OUTPUT = [1.003217200759985, 0.9968055760434815]
const GHNN_OUTPUT_WITH_PARAMETERS = [1.0000350420844089, 0.9999649603746436]

# The parameters of the harmonic oscillator of GeometricProblems, whose three entries the second pin
# was taken with.
const OSCILLATOR_PARAMETERS = (m = 1.0, k = 0.5, ω = sqrt(0.5))

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

# A pinned output holds in `T` against the `Float64` pin to `100 eps(T)` relative, and its element
# type is the precision of the network.
function test_pinned_output(nn, input, pin, T)
    output = nn(input)
    @test eltype(output) === T
    @test isapprox(output, T.(pin); rtol = 100 * eps(T))
end

@testset "Generalized Hamiltonian Neural Network ($T)" for T in REAL_ELTYPES
    # The default constructor: `dim` alone, so `width = dim`, one hidden layer and one integrator.
    nn = NeuralNetwork(GeneralizedHamiltonianArchitecture(2), T; initializer = OneInitializer())
    test_pinned_output(nn, T[1, 1], GHNN_OUTPUT, T)

    # And with the parameters of a system, which lengthen the input of both energy networks.
    nnμ = NeuralNetwork(
        GeneralizedHamiltonianArchitecture(2; parameters = OSCILLATOR_PARAMETERS), T;
        initializer = OneInitializer())
    output = nnμ(T[1, 1], OSCILLATOR_PARAMETERS)
    @test eltype(output) === T
    @test isapprox(output, T.(GHNN_OUTPUT_WITH_PARAMETERS); rtol = 100 * eps(T))
end

# The parameter counts of the thesis (Table 4.2 and Table 7.1). The mapping between its
# networks and this architecture: one Hamiltonian per integrator, one energy network each; the thesis
# counts the first dense layer among the hidden layers, so `nhidden` is one less than its number; the
# output layer is `Linear(width, 1; use_bias = false)`; and the parameters of the system have one
# entry for the pendulum and two for the double pendulum.
@testset "the parameter counts of the thesis" begin
    pendulum = (m = 1.0,)
    double_pendulum = (m = 1.0, ω = 2.0)
    ghnn(dim, width, nhidden, n_integrators,
        parameters = NullParameters()) = parameterlength(
        NeuralNetwork(GeneralizedHamiltonianArchitecture(
        dim; width, nhidden, n_integrators,
        parameters)))

    # 5 Hamiltonians, 2 hidden layers of width 25: the pendulum, the double pendulum and the 3-body
    # problem, whose phase space has 12 dimensions.
    @test ghnn(2, 25, 1, 5) == 7250
    @test ghnn(4, 25, 1, 5) == 7500
    @test ghnn(12, 25, 1, 5) == 8500

    # 10 Hamiltonians, 1 hidden layer of width 50: the pendulum and the double pendulum, whose
    # numbers fix one and two entries of μ.
    @test ghnn(2, 50, 0, 10, pendulum) == 4000
    @test ghnn(4, 50, 0, 10, double_pendulum) == 6000

    # 5 Hamiltonians, 2 hidden layers of width 25, which check the same two values of `|μ|`.
    @test ghnn(2, 25, 1, 5, pendulum) == 7500
    @test ghnn(4, 25, 1, 5, double_pendulum) == 8000
end

@testset "the constructor refuses what it cannot build" begin
    # An odd dimension has no split into `q` and `p`, and the message names it.
    @test_throws ArgumentError GeneralizedHamiltonianArchitecture(3)
    @test occursin("3", argument_error_message(() -> GeneralizedHamiltonianArchitecture(3)))

    # A chain of no integrator is the identity, and a network has no negative hidden layer.
    @test_throws ArgumentError GeneralizedHamiltonianArchitecture(2; n_integrators = 0)
    @test occursin("0", argument_error_message(
        () -> GeneralizedHamiltonianArchitecture(2; n_integrators = 0)))
    @test_throws ArgumentError GeneralizedHamiltonianArchitecture(2; nhidden = -1)
    @test occursin("-1", argument_error_message(
        () -> GeneralizedHamiltonianArchitecture(2; nhidden = -1)))

    # A layer of width zero has no output, and the refusal is at construction.
    for width in (0, -2)
        @test_throws ArgumentError GeneralizedHamiltonianArchitecture(2; width)
        message = argument_error_message(() -> GeneralizedHamiltonianArchitecture(2; width))
        @test occursin("`width` is $(width)", message)
    end

    # `nhidden = 0` is the thesis's one hidden layer, and it builds.
    @test parameterlength(NeuralNetwork(GeneralizedHamiltonianArchitecture(2; nhidden = 0))) >
          0

    # An integer of another width builds the same network.
    narrow = GeneralizedHamiltonianArchitecture(Int32(2))
    @test narrow.dim === 2
    @test parameterlength(NeuralNetwork(narrow)) ==
          parameterlength(NeuralNetwork(GeneralizedHamiltonianArchitecture(2)))
end

@testset "a network without system parameters: `(;)` and `NullParameters()` ($T)" for T in REAL_ELTYPES
    empty_parameters = NeuralNetwork(
        GeneralizedHamiltonianArchitecture(2; parameters = (;)), T;
        initializer = OneInitializer())
    null_parameters = NeuralNetwork(
        GeneralizedHamiltonianArchitecture(2; parameters = NullParameters()), T;
        initializer = OneInitializer())
    x = T[0.4, 0.6]
    @test parameterlength(empty_parameters) == parameterlength(null_parameters)
    @test empty_parameters(x) == null_parameters(x)
    @test empty_parameters(x, (;)) == null_parameters(x, NullParameters())
    @test eltype(empty_parameters(x)) === T
end

@testset "a batch without system parameters keeps its shape ($T)" for T in REAL_ELTYPES
    nn = NeuralNetwork(GeneralizedHamiltonianArchitecture(4; width = 3), T)
    x = rand(T, 4)
    matrix = rand(T, 4, 3)
    tensor = rand(T, 4, 2, 3)
    @test size(nn(x)) == size(x)
    @test size(nn(matrix)) == size(matrix)
    @test size(nn(tensor)) == size(tensor)
    @test eltype(nn(matrix)) === T
    @test eltype(nn(tensor)) === T

    # Every column is a sample of its own: the matrix call is the vector calls side by side, and not
    # the first of them.
    @test all(nn(matrix)[:, i] ≈ nn(matrix[:, i]) for i in axes(matrix, 2))
    @test any(nn(matrix)[:, 1] != nn(matrix)[:, 2])
    @test all(nn(tensor)[:, 1, j] ≈ nn(tensor[:, 1, j]) for j in axes(tensor, 3))
end

@testset "the functor takes the parameters of the system ($T)" for T in REAL_ELTYPES
    architecture = GeneralizedHamiltonianArchitecture(2; parameters = (m = 1.0,))
    nn = NeuralNetwork(architecture, T)
    μ = (m = 2.0,)
    x = rand(T, 2)
    output = nn(x, μ)
    @test output ≈ apply_parametric(nn.model, x, μ, params(nn))
    @test eltype(output) === T

    # A matrix with one μ is not a form `apply_parametric` takes, and the functor does not invent
    # one.
    @test_throws MethodError nn(rand(T, 2, 1), μ)

    # One μ per column of a batch, each entering its own column.
    μs = [(m = 1.0,), (m = 2.0,), (m = 3.0,)]
    batch = rand(T, 2, 1, 3)
    @test size(nn(batch, μs)) == size(batch)
    @test !(nn(batch, μs)[:, :, 1] ≈ nn(batch, μs)[:, :, 2])
    @test all(nn(batch, μs)[:, :, i] ≈ nn(batch[:, :, i:i], [μs[i]])[:, :, 1]
    for i in axes(batch, 3))

    # `apply_parametric` takes any `AbstractVector` of per-sample parameters, and the functor takes
    # the forms it takes: a vector of `NullParameters` and a `Vector{Any}` of `NamedTuple`s, which
    # is not a `Vector{<:NamedTuple}`, reach it too.
    plain = NeuralNetwork(GeneralizedHamiltonianArchitecture(2), T)
    plain_batch = rand(T, 2, 1, 2)
    nulls = [NullParameters(), NullParameters()]
    @test plain(plain_batch, nulls) ≈
          apply_parametric(plain.model, plain_batch, nulls, params(plain))
    untyped = Any[(m = 1.0,), (m = 2.0,)]
    @test nn(plain_batch, untyped) ≈
          apply_parametric(nn.model, plain_batch, untyped, params(nn))
end

# A step hands its energy network the half of the state it differentiates with the flattened
# parameters of the system below it. The generated gradient function reads the entries it was built
# for and ignores the rest, so a `μ` of the wrong length would be silently truncated: the step
# refuses it and names both lengths.
@testset "a step refuses an energy input of the wrong length ($T)" for T in REAL_ELTYPES
    dimension, width, nhidden = 2, 3, 1
    μ = (m = 1.0,)
    layer = SymplecticEulerA(
        SymbolicKineticEnergy(dimension, width, nhidden, tanh; parameters = μ);
        return_parameters = false)
    ps = params(NeuralNetwork(Chain(layer), T)).L1
    z = (q = rand(T, dimension ÷ 2), p = rand(T, dimension ÷ 2))

    # the state has one row and the energy network takes two: the state and the one entry of `μ`
    step = layer(z, μ, ps)
    @test step isa NamedTuple{(:q, :p)}
    @test eltype(step.q) === T

    # one entry more in `μ` is one row more in the input of the energy network
    message = argument_error_message(() -> layer(z, (m = 1.0, ω = 2.0), ps))
    @test occursin("takes 2 inputs", message)
    @test occursin("has 3", message)
end

# A matrix of samples takes one `μ` per column, which is how the layerwise pullback hands a batch to
# a step: column `i` of the result is the step of column `i` with `μs[i]`.
@testset "a step takes one μ per column of a matrix ($T)" for T in REAL_ELTYPES
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0), (m = 3.0, ω = 1.0)]
    states = rand(T, 4, 3)
    for (Layer, energy) in ((SymplecticEulerA, SymbolicKineticEnergy),
        (SymplecticEulerB, SymbolicPotentialEnergy))
        layer = Layer(energy(4, 3, 1, tanh; parameters = first(μs)); return_parameters = true)
        ps = params(NeuralNetwork(Chain(layer), T)).L1
        output, passed_on = @inferred layer((states, μs), ps)
        columns = reduce(hcat, [first(layer(states[:, i], μs[i], ps)) for i in 1:3])
        @test eltype(output) === T
        @test output ≈ columns rtol = 8 * eps(T)
        @test passed_on === μs
        # the three values of `μ` are distinct, so a step that used one of them for every column
        # would give another result
        @test !(output ≈ first(layer(states, first(μs), ps)))
    end
end

# The two steps of one integrator, against the gradient of their own energies, which the test takes
# with `Zygote` rather than from the generated function the layer holds.
@testset "a symplectic-Euler step is the gradient of its energy ($T)" for T in REAL_ELTYPES
    μ = (m = 1.0,)
    dimension, width, nhidden = 2, 3, 1
    half = dimension ÷ 2
    z = (q = rand(T, half), p = rand(T, half))

    for (Layer, energy, sign) in ((SymplecticEulerA, SymbolicKineticEnergy, +1),
        (SymplecticEulerB, SymbolicPotentialEnergy, -1))
        layer = Layer(energy(dimension, width, nhidden, tanh; parameters = μ);
            return_parameters = false)
        nn = NeuralNetwork(Chain(layer), T)
        ps = params(nn).L1
        differentiated = Layer === SymplecticEulerA ? z.p : z.q
        gradient = Zygote.gradient(
            input -> only(layer.energy_model(input, ps)), vcat(differentiated, T[μ.m]))[1]
        output = layer(z, μ, ps)

        # The step changes one half by the gradient of its energy, and leaves the other one alone.
        moved = Layer === SymplecticEulerA ? output.q : output.p
        base = Layer === SymplecticEulerA ? z.q : z.p
        @test moved ≈ base .+ sign .* gradient[1:half]
        @test (Layer === SymplecticEulerA ? output.p : output.q) == differentiated
        @test eltype(moved) === T
    end
end

# The forward pass of a layer is inferred on every shape it takes, which it is not if it holds its
# energy network or its gradient function behind an abstract field.
@testset "the forward pass of a layer is inferred" begin
    μ = (m = 1.0, ω = 2.0)
    for (Layer, energy) in ((SymplecticEulerA, SymbolicKineticEnergy),
        (SymplecticEulerB, SymbolicPotentialEnergy))
        layer = Layer(energy(4, 3, 1, tanh; parameters = μ); return_parameters = false)
        nn = NeuralNetwork(Chain(layer), Float64)
        ps = params(nn).L1
        # Both fields are declared with the layer's own type parameter rather than with an abstract
        # supertype, so that inference sees the network's and the gradient function's type behind
        # them.
        @test fieldtype(typeof(layer), :gradient_function) ===
              typeof(layer.gradient_function)
        @test fieldtype(typeof(layer), :energy_model) === typeof(layer.energy_model)
        z = (q = rand(2), p = rand(2))
        @test @inferred(layer(z, μ, ps)) isa NamedTuple{(:q, :p)}
        @test @inferred(layer(rand(4), μ, ps)) isa Vector{Float64}
        @test @inferred(layer(rand(4, 3), μ, ps)) isa Matrix{Float64}
        @test @inferred(layer(rand(4, 2, 3), μ, ps)) isa Array{Float64, 3}

        # A layer without system parameters takes the state alone.
        plain = Layer(energy(4, 3, 1, tanh); return_parameters = false)
        plain_parameters = params(NeuralNetwork(Chain(plain), Float64)).L1
        @test @inferred(plain(z, plain_parameters)) isa NamedTuple{(:q, :p)}
        @test @inferred(plain(rand(4), plain_parameters)) isa Vector{Float64}
        @test @inferred(plain(rand(4, 3), plain_parameters)) isa Matrix{Float64}
    end
end
