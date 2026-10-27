"""
    SymbolicPullback(arch::HamiltonianArchitecture)

Make a `SymbolicPullback` of [`HNNLoss`](@ref) for a [`HamiltonianArchitecture`](@ref).

Its value is `HNNLoss` on the batch, ``\\ell = \\|R\\|/\\|Y\\|`` with ``R = X_H(X) - Y``. Its
gradient is ``\\Delta \\sum_i \\nabla_\\theta \\tfrac12\\|r_i\\|^2 / (\\|R\\|\\,\\|Y\\|)``: the generated
per-sample gradient of the surrogate ``\\tfrac12\\|r_i\\|^2``, summed over the batch and scaled by the
two norms of the value's residual. The gradient has the parameters' element type, and a zero
residual gives a zero gradient.

`SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used: it sizes the target of the loss
with `output_dimension(nn.model)`, the scalar Hamiltonian, but `HNNLoss` compares against the vector
field, which has the dimension of the input.
"""
function SymbolicPullback(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    hvf = symbolic_hamiltonian_vector_field(nn)
    # `inplace = false`, as in `hamiltonian_vector_field`: `HNNLoss` is also differentiated by `Zygote`.
    loss = HNNLoss(SymbolicNeuralNetworks.build_nn_function(
        hvf, nn.params, nn.input; inplace = false))
    soutput = Symbolics.variables(:y, 1:input_dimension(nn.model))
    gradient = SymbolicNeuralNetworks.symbolic_parameter_gradient(
        sum(abs2, hvf - soutput) / 2, nn)
    gradient_function = SymbolicNeuralNetworks.build_nn_function(
        gradient, nn.params, nn.input, soutput; reduce = +)
    SymbolicPullback(loss, _HNNParameterGradient(gradient_function))
end

# The `fun` of the HNN pullback; it holds the generated gradient and selects the call method below.
struct _HNNParameterGradient{FT}
    gradient_function::FT
end

# More specific than `SymbolicNeuralNetworks`' call, which computes the value and the gradient
# separately and would evaluate `hvf` twice: narrower in the pullback and the parameters, and the
# same in the data.
function (pb::SymbolicPullback{<:HNNLoss, <:_HNNParameterGradient})(
        ps::NetworkParameters, model,
        (input, output)::Tuple{<:ArrayOrNamedTuple, <:ArrayOrNamedTuple})
    norm_R = norm(pb.loss.hvf(input, ps) - output)
    norm_Y = norm(output)
    # `pinv`: a zero residual gives the zero gradient, not `NaN`. Each norm divides each entry on its
    # own, because `‖R‖‖Y‖` overflows, and its inverse underflows, where neither norm does.
    inv_R = pinv(norm_R)
    TP = parameter_eltype(ps)
    gradient_function = pb.fun.gradient_function
    norm_R / norm_Y,
    Δ -> mapstorage(x -> TP.(x .* Δ .* inv_R ./ norm_Y), gradient_function(input, output, ps))
end

# A generalized Hamiltonian network is a map of the state and the parameters of the system, not a
# scalar Hamiltonian, so `HNNLoss` — and with it this pullback — has no value to compare a vector
# field against.
function SymbolicPullback(
        arch::Union{
        GeneralizedHamiltonianArchitecture, ForcedGeneralizedHamiltonianArchitecture})
    throw(ArgumentError(
        "a `$(nameof(typeof(arch)))` is a map of the state and the parameters of the system, not a " *
        "scalar Hamiltonian, so it has no `HNNLoss` pullback. Build the parametric pullback " *
        "instead, `SymbolicPullback(arch, ParametricLoss())`, or train through " *
        "`ZygotePullback(ParametricLoss())`, which is what `train!` uses."))
end

# How many `SymplecticEuler` layers a chain stacks: only the generalized architectures have more than
# one, and a network built around a `Chain` of its own says nothing about how many it holds.
_n_symplectic_integrators(::Any) = 1
function _n_symplectic_integrators(
        arch::Union{
        GeneralizedHamiltonianArchitecture, ForcedGeneralizedHamiltonianArchitecture})
    arch.n_integrators
end

@doc raw"""
    _check_symbolic_pullback_is_tractable(arch)

Throw if one symbolic expression for a generalized Hamiltonian network with `arch`'s integrators
does not finish building.

The pullback traces the whole chain, and a [`SymplecticEuler`](@ref) step inlines the generated
gradient of its energy network — an expression that is itself already a derivative. Stacking
integrators inlines that expression inside itself, so what a chain of them costs grows
multiplicatively rather than additively: one integrator of a small network builds in under a
second, and the expression of two does not finish building.
"""
function _check_symbolic_pullback_is_tractable(arch)
    n = _n_symplectic_integrators(arch)
    n > 1 && throw(ArgumentError(
        "cannot build a `SymbolicPullback` for an architecture with `n_integrators` = $(n): each " *
        "step inlines the generated gradient of its energy network, so the symbolic expression grows " *
        "multiplicatively with the number of integrators and does not finish building. Use " *
        "`ZygotePullback(ParametricLoss())`, which is what `train!` takes by default, or a network " *
        "with one integrator."))
    nothing
end

@doc raw"""
    SymbolicPullback(arch, ::ParametricLoss)

Make a `SymbolicPullback` of [`ParametricLoss`](@extref GeometricMachineLearning ParametricLoss) for
a [`GeneralizedHamiltonianArchitecture`](@ref) or a
[`ForcedGeneralizedHamiltonianArchitecture`](@ref), whose forward pass takes the parameters of the
*system*.

Its value is `ParametricLoss` on the batch, and its gradient is the gradient of that value. On a
stacked batch the value is ``\ell = \|R\|/\|Y\|`` with ``R = \mathrm{NN}(X, \mu) - Y``, and its
gradient is ``\Delta \sum_i \nabla_\theta \tfrac12\|r_i\|^2 / (\|R\|\,\|Y\|)``: the generated
per-sample gradient of the surrogate ``\tfrac12\|r_i\|^2``, summed over the batch and scaled by the
two norms. On a `(q, p)` batch the value is ``\ell = (\|R_q\| + \|R_p\|)/(\|Y_q\| + \|Y_p\|)`` — each
half scaled by its own norm, which is what the loss does with a `(q, p)` pair — and the gradient is
``\Delta (\sum_i \nabla_\theta \tfrac12\|r_{q,i}\|^2/\|R_q\| + \sum_i \nabla_\theta
\tfrac12\|r_{p,i}\|^2/\|R_p\|) / (\|Y_q\| + \|Y_p\|)``. The gradient has the parameters' element
type, and a zero residual gives a zero gradient.

The parameters of the system that `arch` holds fix the shape of `\mu`, and with it the length of the
input of the generated function: the entries of `\mu` follow the state of a sample there, and the
symbolic expression splits them off again. Their *values* enter with each call, one per sample.

# The batch

The pullback takes what a
[`ParametricDataLoader`](@extref GeometricMachineLearning ParametricDataLoader) draws: an input and
an output of `2d \times 1 \times k` with `k` values of `\mu`, or a `(q, p)` pair of
`d \times 1 \times k`. A matrix with one `\mu` is not one of the forms `apply_parametric` takes.

The two batch forms are two quantities on one batch, because the loss weighs them differently:
stacked, `q` above `p`, one norm covers both halves; as a pair, each half is scaled by its own. Each
form's value and gradient are those of `ParametricLoss` on that same form.

`SymbolicPullback(arch::HamiltonianArchitecture)` is the pullback of `HNNLoss` and keeps its
signature; a generalized architecture is refused there.
"""
function SymbolicPullback(
        arch::Union{
            GeneralizedHamiltonianArchitecture, ForcedGeneralizedHamiltonianArchitecture},
        ::ParametricLoss; cse::Bool = true, inplace::Bool = true)
    _check_symbolic_pullback_is_tractable(arch)

    symbolic_network = SymbolicNeuralNetwork(arch)
    model = symbolic_network.model
    symbolic_network_parameters = symbolic_network.params
    flat_system_parameters, layout = GeometricMachineLearning._flatten_system_parameters(
        arch.parameters)

    state_dimension = input_dimension(model)
    sinput = Symbolics.variables(:x, 1:(state_dimension + length(flat_system_parameters)))
    soutput = Symbolics.variables(:y, 1:output_dimension(model))
    symbolic_system_parameters = unflatten(layout, sinput[(state_dimension + 1):end])

    output = model(
        (sinput[1:state_dimension], symbolic_system_parameters), symbolic_network_parameters)
    # One surrogate per half of the state, generated together: a `(q, p)` batch scales the gradient
    # of each half by that half's own residual norm, and a stacked batch by the norm of their sum, so
    # the two conventions come out of one function rather than two. The symbolic model returns a
    # stacked state, `q` above `p`, for either architecture.
    half = state_dimension ÷ 2
    residuals = (q = output[1:half] - soutput[1:half],
        p = output[(half + 1):end] - soutput[(half + 1):end])
    gradient = map(
        residual -> SymbolicNeuralNetworks.symbolic_parameter_gradient(
            sum(abs2, residual) / 2, symbolic_network_parameters),
        residuals)
    gradient_function = SymbolicNeuralNetworks.build_nn_function(
        gradient, symbolic_network_parameters, sinput, soutput;
        reduce = +, cse = cse, inplace = inplace)
    SymbolicPullback(ParametricLoss(), _PGHNNParameterGradient(gradient_function))
end

# The `fun` of the parametric pullback; it holds the generated gradient and selects the call methods
# below.
struct _PGHNNParameterGradient{FT}
    gradient_function::FT
end

# The batch as the generated function takes it: one column per sample. The state and the target are
# stacked this way, `q` above `p`, as the data of a `ParametricDataLoader` are whenever they are not
# already stacked.
_sample_columns(x::AbstractArray{<:Any, 3}) = reshape(x, size(x, 1), size(x, 3))
function _sample_columns(x::NamedTupleOfArrays{<:Any, (:q, :p)})
    vcat(_sample_columns(x.q), _sample_columns(x.p))
end

# The input of the generated function, whose columns are the state of a sample with that sample's
# parameters below it. The columns are concatenated side by side, so the result has
# `input_dimension + |μ|` rows and one column per sample.
function _sample_input(input, μs)
    GeometricMachineLearning.concatenate_array_with_parameters(_sample_columns(input), μs)
end

function (pb::SymbolicPullback{<:ParametricLoss, <:_PGHNNParameterGradient})(
        ps::NetworkParameters, model,
        (input, output, μs)::Tuple{<:AbstractArray, <:AbstractArray, <:AbstractVector})
    target = _sample_columns(output)
    norm_R = norm(_sample_columns(apply_parametric(model, input, μs, ps)) - target)
    norm_Y = norm(target)
    # `pinv`: a zero residual gives the zero gradient, not `NaN`. Each norm divides each entry on its
    # own, because `‖R‖‖Y‖` overflows, and its inverse underflows, where neither norm does.
    inv_R = pinv(norm_R)
    TP = parameter_eltype(ps)
    gradient_function = pb.fun.gradient_function
    # One norm over both halves, so the gradients of the two halves add up before they are scaled.
    norm_R / norm_Y,
    Δ -> mapstorage((g_q, g_p) -> TP.((g_q .+ g_p) .* Δ .* inv_R ./ norm_Y),
        gradient_function(_sample_input(input, μs), target, ps)...)
end

function (pb::SymbolicPullback{<:ParametricLoss, <:_PGHNNParameterGradient})(
        ps::NetworkParameters, model,
        (input,
            output,
            μs)::Tuple{<:NamedTupleOfArrays{<:Any, (:q, :p)},
            <:NamedTupleOfArrays{<:Any, (:q, :p)}, <:AbstractVector})
    prediction = apply_parametric(model, input, μs, ps)
    target_q, target_p = _sample_columns(output.q), _sample_columns(output.p)
    norm_R_q = norm(_sample_columns(prediction.q) - target_q)
    norm_R_p = norm(_sample_columns(prediction.p) - target_p)
    norm_Y = norm(target_q) + norm(target_p)
    # `pinv` per half: a zero residual in one half gives the zero gradient there, not `NaN`.
    inv_R_q, inv_R_p = pinv(norm_R_q), pinv(norm_R_p)
    TP = parameter_eltype(ps)
    gradient_function = pb.fun.gradient_function
    # Each half is scaled by its own residual norm, so the two gradients cannot be added first.
    (norm_R_q + norm_R_p) / norm_Y,
    Δ -> mapstorage((g_q, g_p) -> TP.((g_q .* inv_R_q .+ g_p .* inv_R_p) .* Δ ./ norm_Y),
        gradient_function(_sample_input(input, μs), _sample_columns(output), ps)...)
end
