using HamiltonianNeuralNetworks
using HamiltonianNeuralNetworks: _symplectic_euler_evaluation_point,
                                 hamiltonian_vector_field
using GeometricMachineLearning: Adam, Batch, DataLoader, NetworkParameters, NeuralNetwork,
                                TrainingOptimizer, params, train!
using LinearAlgebra
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

# ---------------------------------------------------------------- ambiguity with `NetworkLoss`

@doc raw"""
Each three-argument loss functor `(loss)(ps, input, output)` must not collide with
`AbstractNeuralNetworks`' own `(::NetworkLoss)(::NeuralNetwork, input, output)`. With an untyped
first argument the two are ambiguous -- same argument count, neither more specific -- and
`loss(nn, input, output)` is a `MethodError: ... is ambiguous` rather than a call. Typing the first
parameter `::NetworkParameters` excludes a `NeuralNetwork`, so the call resolves to upstream's
method, which itself forwards to `loss(nn.params, input, output)` -- the two must therefore return
the same number.
"""
function test_loss_of_a_network_matches_loss_of_its_parameters(T::Type)
    hnn_arch = StandardHamiltonianArchitecture(2, 4, 1)
    nn = NeuralNetwork(hnn_arch, T)
    loss = HNNLoss(hnn_arch)
    input, output = randn(T, 2, 5), randn(T, 2, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)
    @test loss(params(nn), input, output) isa T

    se_arch = StandardHamiltonianArchitecture(2, 4, 1)
    nn = NeuralNetwork(se_arch, T)
    loss = SymplecticEulerLoss(se_arch, 0.1)
    input, output = randn(T, 2, 5), randn(T, 2, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)
    @test loss(params(nn), input, output) isa T
end

for T in REAL_ELTYPES
    test_loss_of_a_network_matches_loss_of_its_parameters(T)
end

# ---------------------------------------------------------------- SymplecticEulerLoss

@doc raw"""
The loss must vanish on a trajectory the learned Hamiltonian actually generates.

Take the network's *own* Hamiltonian vector field, step it with symplectic Euler A, and feed the
resulting pair back in: the residual is then zero by construction, so a non-zero value would mean
the loss is not the method it claims to be.

Symplectic Euler A is implicit in ``q``, so the step is taken by fixed-point iteration.
"""
function symplectic_euler_a_step(hvf, ps, state, Δt; iterations = 60)
    n = length(state) ÷ 2
    q, p = state[1:n], state[(n + 1):(2n)]
    q_next = copy(q)
    for _ in 1:iterations
        field = hvf(vcat(q_next, p), ps)
        q_next = q + Δt * field[1:n]
    end
    field = hvf(vcat(q_next, p), ps)
    vcat(q_next, p + Δt * field[(n + 1):(2n)])
end

# The residual of the network's own step is the round-off of the difference quotient: `eps(T)` on
# the states, divided by `Δt`. Over 50 seeds and both precisions it stays below `10 eps(T) / Δt`, so
# the bound `100 eps(T) / Δt` leaves a factor of ten.
function test_symplectic_euler_loss_vanishes_on_its_own_step(
        T::Type, n::Integer = 1; Δt = T(1.0e-3))
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch, T)
    loss = SymplecticEulerLoss(arch, Δt)
    hvf = hamiltonian_vector_field(arch)

    input = reduce(hcat, [randn(T, 2n) for _ in 1:6])
    output = reduce(hcat,
        [symplectic_euler_a_step(hvf, params(nn), input[:, b], Δt) for b in axes(input, 2)])

    @test loss(params(nn), input, output) isa T
    @test loss(params(nn), input, output) < 100 * eps(T) / Δt

    # A trajectory that is *not* the method's own step must not pass, so the assertion above is
    # not vacuous.
    @test loss(params(nn), input, output + T(0.1) * randn(T, size(output))) >
          100 * eps(T) / Δt
end

for T in REAL_ELTYPES
    test_symplectic_euler_loss_vanishes_on_its_own_step(T, 1)
    test_symplectic_euler_loss_vanishes_on_its_own_step(T, 2)
end

"""
The two variants must differ, and each must take its half of the state from the right array.
"""
function test_symplectic_euler_variants(T::Type, n::Integer = 2)
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch, T)
    input = randn(T, 2n, 5)
    output = randn(T, 2n, 5)

    loss_a = SymplecticEulerLoss(arch, 0.1)
    loss_b = SymplecticEulerLoss(arch, 0.1; variant = :B)

    point_a = _symplectic_euler_evaluation_point(loss_a, input, output)
    point_b = _symplectic_euler_evaluation_point(loss_b, input, output)
    @test point_a == vcat(output[1:n, :], input[(n + 1):(2n), :])
    @test point_b == vcat(input[1:n, :], output[(n + 1):(2n), :])
    @test eltype(point_a) === eltype(point_b) === T
    @test loss_a(params(nn), input, output) != loss_b(params(nn), input, output)
    @test loss_a(params(nn), input, output) isa T
    @test loss_b(params(nn), input, output) isa T

    @test_throws ArgumentError SymplecticEulerLoss(arch, 0.1; variant = :C)
end

for T in REAL_ELTYPES
    test_symplectic_euler_variants(T)
end

"""
Both variants must be differentiable with respect to the parameters. Training is the whole point of
a loss, and a residual built on a symbolic vector field is exactly the kind of expression an AD
system can refuse.
"""
function test_symplectic_euler_loss_gradient(T::Type, n::Integer = 1)
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch, T)
    input = randn(T, 2n, 5)
    output = randn(T, 2n, 5)
    for variant in (:A, :B)
        loss = SymplecticEulerLoss(arch, 0.1; variant = variant)
        dp = Zygote.gradient(ps -> loss(ps, input, output), params(nn))[1]
        @test typeof(dp) <: NetworkParameters
        @test keys(dp) == keys(params(nn))
        @test !all(iszero, dp.L1.W)
        @test eltype(dp.L1.W) === T
    end
end

for T in REAL_ELTYPES
    test_symplectic_euler_loss_gradient(T)
end

"""
A timestep of either precision must not change the element type of the loss: a `Float64` timestep
must not widen the loss of a `Float32` network. The timestep is a constant of the method, not data,
so the data's element type is what decides the result's.
"""
function test_timestep_does_not_widen_the_element_type(T::Type, n::Integer = 1)
    hamiltonian = StandardHamiltonianArchitecture(2n, 4, 1)
    ps = params(NeuralNetwork(hamiltonian, T))
    input = randn(T, 2n, 5)
    output = randn(T, 2n, 5)
    @test SymplecticEulerLoss(hamiltonian, 0.1)(ps, input, output) isa T
    @test SymplecticEulerLoss(hamiltonian, 0.1f0)(ps, input, output) isa T
end

for T in REAL_ELTYPES
    test_timestep_does_not_widen_the_element_type(T)
end

function test_symplectic_euler_training(T::Type; n = 1, Δt = T(1.0e-2), n_epochs = 20)
    # The shape of the tests above. The assertion is that training reduces the loss, which does
    # not depend on the width and depth, and a new shape is a fresh `Zygote` specialisation of the
    # whole loss.
    arch = StandardHamiltonianArchitecture(2n, 4, 1)
    nn = NeuralNetwork(arch, T)
    loss = SymplecticEulerLoss(arch, Δt)
    hvf = hamiltonian_vector_field(arch)

    input = reduce(hcat, [randn(T, 2n) for _ in 1:60])
    output = reduce(hcat,
        [symplectic_euler_a_step(hvf, params(nn), input[:, b], Δt) for b in axes(input, 2)])

    nn = NeuralNetwork(arch, T)
    dl = DataLoader(input, output; suppress_info = true)
    loss_array = train!(
        nn, TrainingOptimizer(nn.params; algorithm = Adam()), dl, Batch(10), n_epochs;
        loss, show_progress = false)
    @test eltype(loss_array) === T
    @test loss_array[end] < loss_array[1]
end

for T in REAL_ELTYPES
    test_symplectic_euler_training(T)
end
