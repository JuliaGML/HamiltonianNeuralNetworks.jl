# Check that the loss for phase space data on the manual's Hamiltonian neural network page has the
# signs of SymplecticEulerLoss with `variant = :B`.
#
# Run from the repo root:
#   julia --startup-file=no --project=scripts scripts/verification/symplectic_euler_loss_formula.jl
#
# The page sums the squares of ∂NN/∂qᵢ + s₁ Δpᵢ/Δt and ∂NN/∂pᵢ + s₂ Δqᵢ/Δt at (qᵗ, pᵗ⁺¹), with
# (s₁, s₂) = (+1, -1). For each choice of the signs, the check takes the pairs of states that make
# every one of these terms zero and evaluates the loss on them. The derivatives of the network come
# from central differences, not from its generated vector field. The loss must vanish for the
# displayed signs. The three other choices are the controls: on their pairs it must not vanish.

using GeometricMachineLearning: NeuralNetwork, params
using HamiltonianNeuralNetworks

const Δt = 0.1
const DIMENSION = 4
const SAMPLES = 5

# A central difference with this step has an error of the order of `STEP^2`, about 4e-11.
const STEP = cbrt(eps(Float64))
const TOLERANCE = 1e-8

"""
The gradient of the scalar network `nn` at `z`, by central differences.
"""
function gradient_of_network(nn, z)
    map(eachindex(z)) do i
        offset = [j == i ? STEP : 0.0 for j in eachindex(z)]
        (only(nn(z + offset)) - only(nn(z - offset))) / (2 * STEP)
    end
end

"""
The pairs of states that make each term of the displayed sum zero for the signs `s₁` and `s₂`. The
evaluation points `(qᵗ, pᵗ⁺¹)` are given, and `pᵗ` and `qᵗ⁺¹` follow from them.
"""
function pairs_solving_the_formula(nn, evaluation_points, s₁, s₂)
    n = size(evaluation_points, 1) ÷ 2
    input, output = copy(evaluation_points), copy(evaluation_points)
    for t in axes(evaluation_points, 2)
        ∇ = gradient_of_network(nn, evaluation_points[:, t])
        input[(n + 1):(2n), t] += s₁ * Δt * ∇[1:n]
        output[1:n, t] -= s₂ * Δt * ∇[(n + 1):(2n)]
    end
    input, output
end

const architecture = StandardHamiltonianArchitecture(DIMENSION, 8, 1)
const nn = NeuralNetwork(architecture, Float64)
const loss = SymplecticEulerLoss(architecture, Δt; variant = :B)
const evaluation_points = [sin(1.0 * i * j) for i in 1:DIMENSION, j in 1:SAMPLES]

for (s₁, s₂) in ((+1, -1), (+1, +1), (-1, -1), (-1, +1))
    input, output = pairs_solving_the_formula(nn, evaluation_points, s₁, s₂)
    value = loss(params(nn), input, output)
    println("signs ", (s₁, s₂), ": loss = ", value)
    if (s₁, s₂) == (+1, -1)
        @assert value < TOLERANCE "the displayed signs do not make the loss vanish"
    else
        @assert value > 1e-2 "the control $((s₁, s₂)) makes the loss vanish"
    end
end

println("\nOnly the signs (+1, -1), which the manual displays, make the loss vanish.")
