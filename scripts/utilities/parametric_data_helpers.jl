# The data of a time-dependent problem in the layout `ParametricDataLoader` takes: one sample per
# time step, with the time as the parameter of the system. A `ParametricDataLoader` holds one
# trajectory per entry of its third axis, so each sample below is a trajectory of two time steps,
# its first column the state and its second the target.

using GeometricMachineLearning: ParametricDataLoader

"""
    forced_harmonic_oscillator_solution(t, initial_conditions; omega, Omega, F)

The analytic solution of the sinusoidally forced harmonic oscillator
``\\ddot{q} + \\omega^2 q = F\\sin(\\Omega t)``, as `(q, p)` arrays of size
`(number of initial conditions, length(t))`.
"""
function forced_harmonic_oscillator_solution(
        t::AbstractVector, initial_conditions; omega::Real, Omega::Real, F::Real)
    n = length(initial_conditions)
    q = zeros(n, length(t))
    p = zeros(n, length(t))
    amplitude = F / (omega^2 - Omega^2)
    for (i, time) in enumerate(t), j in 1:n

        q[j, i] = (initial_conditions[j].p - Omega * amplitude) / omega *
                  sin(omega * time) +
                  initial_conditions[j].q * cos(omega * time) +
                  amplitude * sin(Omega * time)
        p[j, i] = -omega * initial_conditions[j].q * sin(omega * time) +
                  (initial_conditions[j].p - Omega * amplitude) * cos(omega * time) +
                  Omega * amplitude * cos(Omega * time)
    end
    (q = q, p = p)
end

"""
    turn_parameters_into_correct_format(t, n_initial_conditions)

One `(t = …,)` per sample of the data `turn_q_p_data_into_correct_format` lays out: the time steps of
the first initial condition, then those of the second, and so on. The last time step of a trajectory
is a target and never a sample.
"""
function turn_parameters_into_correct_format(t::AbstractVector, n_initial_conditions::Integer)
    steps = [(t = time_step,) for time_step in t[1:(end - 1)]]
    reduce(vcat, (copy(steps) for _ in 1:n_initial_conditions))
end

"""
    turn_q_p_data_into_correct_format(q, p)

The `(q, p)` trajectories of `q` and `p`, of size `(number of initial conditions, length(t))`, as a
pair of tensors of size `(1, 2, number of samples)`: one row, the single degree of freedom a
trajectory of the oscillator has, the state of a sample in the first column and its target, one time
step later, in the second.
"""
function turn_q_p_data_into_correct_format(q::AbstractMatrix, p::AbstractMatrix)
    n_initial_conditions, n_time_steps = size(q)
    size(p) == size(q) || throw(ArgumentError(
        "`q` has the size $(size(q)) and `p` has the size $(size(p))."))
    samples = n_time_steps - 1
    q_pairs = Array{eltype(q), 3}(undef, 1, 2, samples * n_initial_conditions)
    p_pairs = similar(q_pairs)
    for i in 1:n_initial_conditions, j in 1:samples

        sample = (i - 1) * samples + j
        q_pairs[:, 1, sample] .= q[i, j]
        q_pairs[:, 2, sample] .= q[i, j + 1]
        p_pairs[:, 1, sample] .= p[i, j]
        p_pairs[:, 2, sample] .= p[i, j + 1]
    end
    (q = q_pairs, p = p_pairs)
end

"""
    load_time_dependent_harmonic_oscillator_with_parametric_data_loader(q, p, t, n_initial_conditions)

A `ParametricDataLoader` of the trajectories `q` and `p`, with the time of each sample as the
parameter of its system.
"""
function load_time_dependent_harmonic_oscillator_with_parametric_data_loader(
        q::AbstractMatrix, p::AbstractMatrix, t::AbstractVector, n_initial_conditions::Integer)
    pairs = turn_q_p_data_into_correct_format(q, p)
    ParametricDataLoader(
        pairs, turn_parameters_into_correct_format(t, n_initial_conditions))
end
