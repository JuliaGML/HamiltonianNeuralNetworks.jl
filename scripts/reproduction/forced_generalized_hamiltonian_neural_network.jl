# Train the three forcings of `ForcedGeneralizedHamiltonianArchitecture` on the sinusoidally forced
# harmonic oscillator, then integrate one trajectory with the learned network and plot it against the
# analytic solution.
#
# Time is the parameter of the system here: the forcing is explicitly time dependent, so the network
# takes `(q, p)` and `t`, and the `ParametricDataLoader` carries one `(t = …,)` per sample.

using HamiltonianNeuralNetworks
using GeometricMachineLearning: Adam, Batch, NeuralNetwork, TrainingOptimizer, train!
using CairoMakie

# the helpers that lay the trajectories out as `ParametricDataLoader` wants them
include("../utilities/parametric_data_helpers.jl")

# `smoke_size(full, smoke)` returns the second when HNN_SMOKE is set. See scripts/README.md.
include("../utilities/smoke.jl")

# natural frequency of the harmonic oscillator
const omega = 1.0

# frequency and amplitude of the external sinusoidal forcing
const Omega = 3.5
const F = 0.9

# number of initial conditions per dimension, so `ni_dim^2` trajectories in total
const ni_dim = smoke_size(10, 2)

# the time grid
const T = 2π * 20
const nt = smoke_size(1000, 20)
const dt = T / nt

const IC = vec([(q = q0, p = p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)])
const t = collect(dt * range(0, nt, step = 1))

const q, p = forced_harmonic_oscillator_solution(t, IC; omega = omega, Omega = Omega, F = F)

const dl = load_time_dependent_harmonic_oscillator_with_parametric_data_loader(
    q, p, t, length(IC))

# The three forcings. `Q`, `P` and `QP` name what the forcing *reads*, not what it changes: all
# three add to `p` and leave `q` alone, which is what a force does to the `ṗ` equation. One
# `NamedTuple` of system parameters is enough for the architecture, which uses it for its shape.
const n_integrators = 2
const system_parameters = (t = t[1],)

function forced_architecture(forcing_type, layer_width)
    ForcedGeneralizedHamiltonianArchitecture(2;
        activation = tanh, width = layer_width, nhidden = 1, n_integrators = n_integrators,
        parameters = system_parameters, forcing_type = forcing_type)
end

const networks = (P = NeuralNetwork(forced_architecture(:P, 1)),
    Q = NeuralNetwork(forced_architecture(:Q, 1)),
    QP = NeuralNetwork(forced_architecture(:QP, 2)))

const batch = Batch(smoke_size(128, 8))
const n_epochs = smoke_size(200, 2)

# `train!` takes a `ZygotePullback` of `ParametricLoss` for a `ParametricDataLoader` by default. The
# symbolic pullback is not an option here: it refuses a chain of more than one integrator, because
# each of its steps inlines the generated gradient of its energy network.
for network in values(networks)
    train!(
        network, TrainingOptimizer(network.params; algorithm = Adam()), dl, batch, n_epochs)
end

# Integrate one trajectory with the `QP` network and compare it against the analytic solution. The
# network is a map over one step, so the parameter of the system — here the time — is passed at each
# step alongside the state.
const trajectory_number = 1
const trajectory = (q = zeros(1, nt + 1), p = zeros(1, nt + 1))
trajectory.q[1, 1] = q[trajectory_number, 1]
trajectory.p[1, 1] = p[trajectory_number, 1]
for step in 1:nt
    output = networks.QP(
        (q = [trajectory.q[1, step]], p = [trajectory.p[1, step]]), (t = t[step],))
    trajectory.q[1, step + 1] = output.q[1]
    trajectory.p[1, step + 1] = output.p[1]
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "time step", ylabel = "q")
lines!(ax, trajectory.q[1, :]; label = "network")
lines!(ax, q[trajectory_number, :]; label = "analytic")
axislegend(ax)

# The script writes no file into the repository: the figure goes to `HNN_RESULTS_DIR`, or to a
# directory of its own under the system's temporary directory, and its path is printed.
const results_dir = get(ENV, "HNN_RESULTS_DIR", mktempdir())
const figure_path = joinpath(results_dir, "forced_generalized_hamiltonian_neural_network.png")
save(figure_path, fig)
println("wrote ", figure_path)
