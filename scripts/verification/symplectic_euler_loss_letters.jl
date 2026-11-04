# Check that the variants of SymplecticEulerLoss carry the letters of Leimkuhler and Reich (2004),
# Eqs. (4.6)-(4.9): `:A` evaluates at (qₙ₊₁, pₙ) and `:B` at (qₙ, pₙ₊₁).
#
# Run from the repo root:
#   julia --startup-file=no --project=. scripts/verification/symplectic_euler_loss_letters.jl
#
# A random network's Hamiltonian is non-separable in general, and the script measures that it is at
# the start point. One step of each formula is solved by fixed-point iteration through the network's
# own vector field. A variant must vanish on the step of its letter. The other pair is the control:
# a variant must not vanish on the step of the other letter.

using GeometricMachineLearning: CPU, NeuralNetwork, params
using HamiltonianNeuralNetworks
using HamiltonianNeuralNetworks: hamiltonian_vector_field
using Printf
import Random

Random.seed!(1234)

arch = StandardHamiltonianArchitecture(2, 16, 2)
nn = NeuralNetwork(arch, CPU(), Float64)
ps = params(nn)
hvf = hamiltonian_vector_field(arch)
X(z) = vec(hvf(reshape(z, 2, 1), ps))

function fixed_point(z₀, h, point)
    z = copy(z₀)
    for _ in 1:500
        znew = z₀ + h * X(point(z₀, z))
        maximum(abs, znew - z) < 1e-14 && return znew
        z = znew
    end
    error("no convergence")
end

euler_a(z₀, h) = fixed_point(z₀, h, (a, b) -> [b[1], a[2]])
euler_b(z₀, h) = fixed_point(z₀, h, (a, b) -> [a[1], b[2]])

z₀, h = [0.3, -0.4], 0.01
zA, zB = euler_a(z₀, h), euler_b(z₀, h)

δ = 1e-5
mixed = (X(z₀ + [0, δ])[2] - X(z₀ - [0, δ])[2]) / (2δ)
println("-H_qp at z₀ = ", mixed, " (zero for a separable H)")
abs(mixed) > 1e-3 || error("the network is separable at z₀, so the check says nothing")

@printf("%-10s %14s %14s\n", "variant", "on Euler-A", "on Euler-B")
function loss(v, z)
    SymplecticEulerLoss(arch, h; variant = v)(ps, reshape(z₀, 2, 1), reshape(z, 2, 1))
end
for v in (:A, :B)
    @printf("%-10s %14.3e %14.3e\n", v, loss(v, zA), loss(v, zB))
end

maximum((loss(:A, zA), loss(:B, zB))) < 1e-12 || error("a variant misses its letter")
minimum((loss(:A, zB), loss(:B, zA))) > 1e-4 || error("a variant reaches the other letter")

println("COMPLETED")
