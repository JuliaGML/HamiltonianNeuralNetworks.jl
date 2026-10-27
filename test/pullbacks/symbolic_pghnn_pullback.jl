# The symbolic pullback of a network that takes the parameters of the system: its value, its
# gradient against `Zygote`'s gradient of the same loss on the same batch, and the edges that make it
# a different quantity from a sum of per-sample relative gradients.

using HamiltonianNeuralNetworks
using GeometricMachineLearning: NeuralNetwork, ParametricLoss, ZygotePullback,
                                ParametricDataLoader
using GeometricBase: NullParameters
import AbstractNeuralNetworks
using AbstractNeuralNetworks: OneInitializer, params
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: norm
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/parameter_leaves.jl")

Random.seed!(1234)

const SYSTEM_PARAMETERS = (m = 1.0, ω = 2.0)

# A single integrator: the smallest parametric network whose symbolic expression still finishes
# building.
function single_integrator(parameters = SYSTEM_PARAMETERS)
    GeneralizedHamiltonianArchitecture(2;
        width = 2, nhidden = 1, n_integrators = 1, parameters)
end

# The value of the pullback is `ParametricLoss` on the batch and its gradient is the gradient of that
# value. The reference is `Zygote`'s gradient of the same loss on the same batch, whose batch holds
# more than one sample with distinct values of `μ`: the sum of per-sample *relative* gradients is a
# different quantity from there on.
@testset "Symbolic pullback of a single-integrator PGHNN ($T)" for T in REAL_ELTYPES
    n_samples = 4
    architecture = single_integrator()
    nn = NeuralNetwork(architecture, T)
    pullback = SymbolicPullback(architecture, ParametricLoss())
    input = rand(T, 2, 1, n_samples)
    output = rand(T, 2, 1, n_samples)
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0), (m = 3.0, ω = 1.0), (m = 0.5, ω = 2.5)]

    value, back = pullback(params(nn), nn.model, (input, output, μs))
    gradient = back(one(value))

    @test value isa T
    @test value ≈ ParametricLoss()(nn.model, params(nn), input, output, μs)
    @test gradient isa NetworkParameters
    @test keys(gradient) == keys(params(nn))

    reference_value, reference_back = ZygotePullback(ParametricLoss())(
        params(nn), nn.model, (input, output, μs))
    @test value ≈ reference_value rtol = T(√eps(T))
    foreach_parameter(gradient, reference_back(one(reference_value))) do leaf, reference
        @test eltype(leaf) === T
        @test leaf ≈ reference rtol = 1000 * eps(T)
    end

    # A gradient of all zeros would pass the comparison above if the reference were zero too, so the
    # network has to depend on its parameters here.
    largest = Ref(zero(T))
    foreach_parameter(gradient) do leaf
        largest[] = max(largest[], maximum(abs, leaf))
    end
    @test largest[] > sqrt(eps(T))
end

# A `ParametricDataLoader` draws one of two batch forms, and the pullback takes both: a
# `2d × 1 × k` tensor, or a `(q, p)` pair of `d × 1 × k` tensors. Each form's value and gradient are
# those of `ParametricLoss` on that same form, and the loss weighs the two halves of a pair by their
# own residual norms, so the two forms are two quantities on one batch.
@testset "the batch forms of a ParametricDataLoader ($T)" for T in REAL_ELTYPES
    n_samples = 3
    architecture = single_integrator()
    nn = NeuralNetwork(architecture, T)
    pullback = SymbolicPullback(architecture, ParametricLoss())
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0), (m = 3.0, ω = 1.0)]
    input = rand(T, 2, 1, n_samples)
    output = rand(T, 2, 1, n_samples)
    @test size(input) == (2, 1, n_samples)

    value, back = pullback(params(nn), nn.model, (input, output, μs))
    @test value ≈ ParametricLoss()(nn.model, params(nn), input, output, μs)
    reference_value, reference_back = ZygotePullback(ParametricLoss())(
        params(nn), nn.model, (input, output, μs))
    @test value ≈ reference_value rtol = T(√eps(T))
    foreach_parameter(back(one(value)), reference_back(one(reference_value))) do leaf,
    reference
        @test eltype(leaf) === T
        @test leaf ≈ reference rtol = 1000 * eps(T)
    end

    q, p = input[1:1, :, :], input[2:2, :, :]
    q_output, p_output = output[1:1, :, :], output[2:2, :, :]
    paired_input = (q = q, p = p)
    paired_output = (q = q_output, p = p_output)
    paired_value, paired_back = pullback(params(nn), nn.model,
        (paired_input, paired_output, μs))
    @test paired_value ≈
          ParametricLoss()(nn.model, params(nn), paired_input, paired_output, μs)
    paired_reference_value, paired_reference_back = ZygotePullback(ParametricLoss())(
        params(nn), nn.model, (paired_input, paired_output, μs))
    @test paired_value ≈ paired_reference_value rtol = T(√eps(T))
    foreach_parameter(paired_back(one(paired_value)),
        paired_reference_back(one(paired_reference_value))) do leaf, reference
        @test eltype(leaf) === T
        @test leaf ≈ reference rtol = 1000 * eps(T)
    end
    # The two conventions are two quantities, and the batch above separates them: one stacked norm
    # for both halves against one norm per half.
    @test !isapprox(paired_value, value, rtol = T(√eps(T)))

    # One sample is a batch of one column, and its value is the loss on that column.
    single_value, single_back = pullback(params(nn), nn.model,
        (input[:, :, 1:1], output[:, :, 1:1], [μs[1]]))
    @test single_value ≈ ParametricLoss()(nn.model, params(nn), input[:, :, 1:1],
        output[:, :, 1:1], [μs[1]])
    @test keys(single_back(one(single_value))) == keys(params(nn))
end

# A `(q, p)` batch whose residual vanishes in both halves has a zero gradient there, not `NaN`: each
# half's residual norm enters through `pinv`.
@testset "a zero residual of a `(q, p)` batch gives a zero gradient ($T)" for T in REAL_ELTYPES
    architecture = single_integrator()
    nn = NeuralNetwork(architecture, T)
    pullback = SymbolicPullback(architecture, ParametricLoss())
    input = rand(T, 2, 1, 2)
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0)]
    paired_input = (q = input[1:1, :, :], p = input[2:2, :, :])
    prediction = HamiltonianNeuralNetworks.apply_parametric(nn.model, paired_input, μs,
        params(nn))
    output = (q = prediction.q, p = prediction.p)
    value, back = pullback(params(nn), nn.model, (paired_input, output, μs))
    @test value isa T
    @test value == 0
    foreach_parameter(back(one(value))) do leaf
        @test eltype(leaf) === T
        @test all(iszero, leaf)
    end
end

# The symbolic input is as long as the flattened parameters of the system, not as its keys: a leaf
# that is a vector contributes one entry per element.
@testset "the symbolic input is sized by the entries of μ ($T)" for T in REAL_ELTYPES
    parameters = (a = [1.0, 2.0],)
    architecture = single_integrator(parameters)
    nn = NeuralNetwork(architecture, T)
    pullback = SymbolicPullback(architecture, ParametricLoss())

    μs = fill((a = [1.0, 2.0],), 3)
    input = rand(T, 2, 1, 3)
    output = rand(T, 2, 1, 3)
    value, back = pullback(params(nn), nn.model, (input, output, μs))
    reference_value, reference_back = ZygotePullback(ParametricLoss())(
        params(nn), nn.model, (input, output, μs))
    @test value isa T
    @test value ≈ reference_value
    foreach_parameter(back(one(value)), reference_back(one(reference_value))) do leaf,
    reference
        @test leaf ≈ reference rtol = 1000 * eps(T)
    end
end

# The gradient has the element type of the parameters, on both batch forms and whatever the
# precision of the seed: a `Float64` seed into a `Float32` network's pullback gives `Float32` leaves.
@testset "a Float64 seed keeps the parameters' element type" begin
    architecture = single_integrator()
    nn = NeuralNetwork(architecture, Float32)
    pullback = SymbolicPullback(architecture, ParametricLoss())
    input = rand(Float32, 2, 1, 2)
    output = rand(Float32, 2, 1, 2)
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0)]
    value, back = pullback(params(nn), nn.model, (input, output, μs))
    @test value isa Float32
    foreach_parameter(back(1.0)) do leaf
        @test eltype(leaf) === Float32
    end

    # The `(q, p)` batch has a call method of its own, with a conversion of its own.
    paired_input = (q = input[1:1, :, :], p = input[2:2, :, :])
    paired_output = (q = output[1:1, :, :], p = output[2:2, :, :])
    paired_value, paired_back = pullback(params(nn), nn.model,
        (paired_input, paired_output, μs))
    @test paired_value isa Float32
    foreach_parameter(paired_back(1.0)) do leaf
        @test eltype(leaf) === Float32
    end
end

# A zero residual gives a zero gradient rather than a `NaN`: the two norms enter through `pinv`.
@testset "a zero residual gives a zero gradient ($T)" for T in REAL_ELTYPES
    architecture = single_integrator()
    nn = NeuralNetwork(architecture, T)
    pullback = SymbolicPullback(architecture, ParametricLoss())
    input = rand(T, 2, 1, 2)
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0)]
    output = HamiltonianNeuralNetworks.apply_parametric(nn.model, input, μs, params(nn))
    value, back = pullback(params(nn), nn.model, (input, output, μs))
    @test value isa T
    @test value == 0
    foreach_parameter(back(one(value))) do leaf
        @test eltype(leaf) === T
        @test all(iszero, leaf)
    end
end

# A network without system parameters builds one pullback from `(;)` and from `NullParameters()`.
@testset "`(;)` and `NullParameters()` build the same pullback ($T)" for T in REAL_ELTYPES
    empty_architecture = GeneralizedHamiltonianArchitecture(2; parameters = (;))
    null_architecture = GeneralizedHamiltonianArchitecture(2; parameters = NullParameters())
    empty_network = NeuralNetwork(empty_architecture, T; initializer = OneInitializer())
    null_network = NeuralNetwork(null_architecture, T; initializer = OneInitializer())
    empty_parameters = SymbolicPullback(empty_architecture, ParametricLoss())
    null_parameters = SymbolicPullback(null_architecture, ParametricLoss())

    input = rand(T, 2, 1, 2)
    output = rand(T, 2, 1, 2)
    value, back = empty_parameters(params(empty_network), empty_network.model,
        (input, output, [(;), (;)]))
    null_value, null_back = null_parameters(params(null_network), null_network.model,
        (input, output, [NullParameters(), NullParameters()]))
    @test value isa T
    @test value == null_value
    foreach_parameter(back(one(value)), null_back(one(null_value))) do leaf, reference
        @test eltype(leaf) === T
        @test leaf == reference
    end
end

# A chain of two integrators inlines the generated gradient of each energy network inside the next
# one, so one expression for the chain holds a copy of each of them per integrator, each inside the
# next. The pullback refuses such a chain rather than building it, and the message names the
# parameter and the pullback to use instead.
@testset "the pullback refuses a stacked architecture" begin
    stacked = GeneralizedHamiltonianArchitecture(2; n_integrators = 2,
        parameters = SYSTEM_PARAMETERS)
    @test_throws ArgumentError SymbolicPullback(stacked, ParametricLoss())
    message = try
        SymbolicPullback(stacked, ParametricLoss())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("n_integrators", message)
    @test occursin("ZygotePullback", message)

    # the forced architecture is counted by its own integrators, and refused for the same reason
    forced = ForcedGeneralizedHamiltonianArchitecture(2; n_integrators = 2,
        parameters = SYSTEM_PARAMETERS)
    @test_throws ArgumentError SymbolicPullback(forced, ParametricLoss())
    forced_message = try
        SymbolicPullback(forced, ParametricLoss())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("n_integrators", forced_message)
    @test occursin("ZygotePullback", forced_message)

    # one integrator is the supported case, and it builds
    @test SymbolicPullback(single_integrator(), ParametricLoss()) isa SymbolicPullback
    @test SymbolicPullback(
        ForcedGeneralizedHamiltonianArchitecture(2;
            parameters = SYSTEM_PARAMETERS),
        ParametricLoss()) isa SymbolicPullback
end

# A forced architecture's pullback is the pullback of `ParametricLoss` as a plain one's is: its value
# and its gradient are those of the loss on both forms of a batch, with the forcing layers in the
# chain that the gradient goes through.
@testset "the pullback of a forced architecture ($T)" for T in REAL_ELTYPES
    n_samples = 2
    architecture = ForcedGeneralizedHamiltonianArchitecture(4;
        width = 2, nhidden = 1, n_forcing_layers = 1, forcing_type = :P,
        parameters = SYSTEM_PARAMETERS)
    nn = NeuralNetwork(architecture, T)
    pullback = SymbolicPullback(architecture, ParametricLoss())
    μs = [(m = 1.0, ω = 2.0), (m = 2.0, ω = 3.0)]
    input = rand(T, 4, 1, n_samples)
    output = rand(T, 4, 1, n_samples)

    value, back = pullback(params(nn), nn.model, (input, output, μs))
    @test value isa T
    @test value ≈ ParametricLoss()(nn.model, params(nn), input, output, μs)
    reference_value, reference_back = ZygotePullback(ParametricLoss())(
        params(nn), nn.model, (input, output, μs))
    @test value ≈ reference_value rtol = T(√eps(T))
    foreach_parameter(back(one(value)), reference_back(one(reference_value))) do leaf,
    reference
        @test eltype(leaf) === T
        @test leaf ≈ reference rtol = 1000 * eps(T)
    end

    paired_input = (q = input[1:2, :, :], p = input[3:4, :, :])
    paired_output = (q = output[1:2, :, :], p = output[3:4, :, :])
    paired_value, paired_back = pullback(params(nn), nn.model,
        (paired_input, paired_output, μs))
    @test paired_value isa T
    @test paired_value ≈
          ParametricLoss()(nn.model, params(nn), paired_input, paired_output, μs)
    paired_reference_value, paired_reference_back = ZygotePullback(ParametricLoss())(
        params(nn), nn.model, (paired_input, paired_output, μs))
    @test paired_value ≈ paired_reference_value rtol = T(√eps(T))
    foreach_parameter(paired_back(one(paired_value)),
        paired_reference_back(one(paired_reference_value))) do leaf, reference
        @test eltype(leaf) === T
        @test leaf ≈ reference rtol = 1000 * eps(T)
    end
end

# A generalized Hamiltonian network is a map rather than a scalar Hamiltonian, so the pullback of
# `HNNLoss` refuses it and names the parametric form.
@testset "a generalized architecture has no `HNNLoss` pullback" begin
    for architecture in (GeneralizedHamiltonianArchitecture(2),
        ForcedGeneralizedHamiltonianArchitecture(2))
        @test_throws ArgumentError SymbolicPullback(architecture)
        message = try
            SymbolicPullback(architecture)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("ParametricLoss", message)
    end
    @test SymbolicPullback(
        StandardHamiltonianArchitecture(2, 2, 1, AbstractNeuralNetworks.TanhActivation())) isa
          SymbolicPullback
end
