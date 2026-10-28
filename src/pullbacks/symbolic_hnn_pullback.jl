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

@doc raw"""
    SymbolicPullback(arch, ::ParametricLoss)

Make a `SymbolicPullback` of [`ParametricLoss`](@extref GeometricMachineLearning.ParametricLoss) for
a [`GeneralizedHamiltonianArchitecture`](@ref) or a
[`ForcedGeneralizedHamiltonianArchitecture`](@ref), whose forward pass takes the parameters of the
*system*. `train!` takes it as its `loss`.

Its value is `ParametricLoss` on the batch, and its gradient is the gradient of that value. On a
stacked batch the value is ``\ell = \|R\|/\|Y\|`` with ``R = \mathrm{NN}(X, \mu) - Y``, and on a
`(q, p)` batch it is ``\ell = (\|R_q\| + \|R_p\|)/(\|Y_q\| + \|Y_p\|)`` — each half scaled by its own
norm, which is what the loss does with a `(q, p)` pair. The gradient has the parameters' element
type, and a zero residual gives a zero gradient.

# The batch

The pullback takes what a
[`ParametricDataLoader`](@extref GeometricMachineLearning.ParametricDataLoader) draws: an input and
an output of $2d \times 1 \times k$ with $k$ values of $\mu$, one per sample, or a `(q, p)` pair of
$d \times 1 \times k$. A matrix with one $\mu$ is not one of the forms `apply_parametric` takes.

# The construction

The parameters of the system that `arch` holds fix the shape of $\mu$; their *values* enter with each
call, one per sample.

For a `GeneralizedHamiltonianArchitecture` the pullback is composed layer by layer, by the layerwise
sweep of `SymbolicNeuralNetworks`: each [`SymplecticEuler`](@ref) step gets generated code of its
own, and the parameters of the system cross between two steps beside the state. So the expression a
step is built from holds that step alone, and any number of integrators builds. At a call the sweep
is seeded with the cotangent of the loss at the output — the residual of each half, scaled by the two
norms the value divides by — so the gradient of the batch comes out of one backward pass.

For a `ForcedGeneralizedHamiltonianArchitecture` the sweep does not apply, because the
[`ForcingLayer`](@extref GeometricMachineLearning.ForcingLayer) of its chain declares no seam for
it. The pullback is then built from one expression for the whole chain, the generated per-sample
gradient of ``\tfrac12\|r_i\|^2`` for each half, which is scaled at a call the same way. That
expression holds one copy of each step's generated gradient inside the next, so it grows
multiplicatively with the number of integrators, and a forced architecture with `n_integrators > 1`
is an `ArgumentError`; `ZygotePullback(ParametricLoss())`, which is what `train!` takes by default,
trains it.

`SymbolicPullback(arch::HamiltonianArchitecture)` is the pullback of `HNNLoss` and keeps its
signature; a generalized architecture is refused there.
"""
function SymbolicPullback(
        arch::Union{
            GeneralizedHamiltonianArchitecture, ForcedGeneralizedHamiltonianArchitecture},
        ::ParametricLoss; cse::Bool = true, inplace::Bool = true)
    SymbolicPullback(ParametricLoss(), _parametric_gradient(arch; cse, inplace))
end

# The stand-in loss the layerwise sweep is built for. Its seed at the output is the residual
# `ŷ - y`; a call of the pullback replaces that seed by the residual scaled to the cotangent of
# `ParametricLoss`, which depends on the norms of the whole batch.
struct _HalfSquaredResidual <: NetworkLoss end

function SymbolicNeuralNetworks.loss_expression(::_HalfSquaredResidual, ŷ, y)
    sum(abs2, ŷ - y) / 2
end

# The `fun` of the layerwise pullback; it holds the sweep and selects the call method below.
struct _LayerwiseParameterGradient{GT}
    gradient_function::GT
end

function _parametric_gradient(arch::GeneralizedHamiltonianArchitecture; cse, inplace)
    pullback = SymbolicNeuralNetworks.SymbolicPullback(
        SymbolicNeuralNetwork(arch), _HalfSquaredResidual(); layerwise = true, cse, inplace)
    _LayerwiseParameterGradient(pullback.fun.gradient_function)
end

# The sweep with another seed: the generated code of the steps does not depend on the loss.
function _reseeded(sweep::SymbolicNeuralNetworks.LayerwiseGradientFunction{Keys},
        seed) where {Keys}
    SymbolicNeuralNetworks.LayerwiseGradientFunction{Keys}(sweep.steps, seed)
end

# The `fun` of the pullback built from one expression; it holds the generated gradient of the two
# halves and selects the call method below.
struct _PGHNNParameterGradient{FT}
    gradient_function::FT
end

function _parametric_gradient(
        arch::ForcedGeneralizedHamiltonianArchitecture; cse, inplace)
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
    # stacked state, `q` above `p`.
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
    _PGHNNParameterGradient(gradient_function)
end

# The expression of a forced chain holds one copy of each step's generated gradient inside the next,
# and the layerwise sweep, which would avoid that, cannot compose the chain past its `ForcingLayer`.
function _check_symbolic_pullback_is_tractable(arch::ForcedGeneralizedHamiltonianArchitecture)
    n = arch.n_integrators
    n > 1 && throw(ArgumentError(
        "cannot build a `SymbolicPullback` for a `ForcedGeneralizedHamiltonianArchitecture` with " *
        "`n_integrators` = $(n): its pullback is one expression for the whole chain, which holds " *
        "one copy of each step's generated gradient inside the next, and the layerwise sweep of " *
        "`SymbolicNeuralNetworks` cannot compose the chain past its `ForcingLayer`. Use " *
        "`ZygotePullback(ParametricLoss())`, which is what `train!` takes by default, or a network " *
        "with one integrator."))
    nothing
end

# The batch forms of a `ParametricDataLoader`: a stacked `2d × 1 × k` input and output, or a `(q, p)`
# pair of `d × 1 × k` tensors, each with `k` values of `μ`.
const _ParametricBatch = Union{
    Tuple{<:AbstractArray, <:AbstractArray, <:AbstractVector},
    Tuple{<:NamedTupleOfArrays{<:Any, (:q, :p)}, <:NamedTupleOfArrays{<:Any, (:q, :p)},
        <:AbstractVector}}

# The batch as the generated code takes it: one column per sample. The state and the target are
# stacked this way, `q` above `p`, as the data of a `ParametricDataLoader` are whenever they are not
# already stacked.
_sample_columns(x::AbstractArray{<:Any, 3}) = reshape(x, size(x, 1), size(x, 3))
function _sample_columns(x::NamedTupleOfArrays{<:Any, (:q, :p)})
    vcat(_sample_columns(x.q), _sample_columns(x.p))
end

# The value of `ParametricLoss` on a batch, and what its gradient is scaled by: the inverse residual
# norm of each half of the state and the norm of the target. `pinv`: a zero residual gives the zero
# gradient, not `NaN`. A stacked batch has one residual norm over both halves.
function _value_and_scales(model, ps, input::AbstractArray, output::AbstractArray, μs)
    target = _sample_columns(output)
    norm_R = norm(_sample_columns(apply_parametric(model, input, μs, ps)) - target)
    norm_Y = norm(target)
    norm_R / norm_Y, pinv(norm_R), pinv(norm_R), norm_Y
end

function _value_and_scales(model, ps, input::NamedTupleOfArrays{<:Any, (:q, :p)},
        output::NamedTupleOfArrays{<:Any, (:q, :p)}, μs)
    prediction = apply_parametric(model, input, μs, ps)
    target_q, target_p = _sample_columns(output.q), _sample_columns(output.p)
    norm_R_q = norm(_sample_columns(prediction.q) - target_q)
    norm_R_p = norm(_sample_columns(prediction.p) - target_p)
    norm_Y = norm(target_q) + norm(target_p)
    (norm_R_q + norm_R_p) / norm_Y, pinv(norm_R_q), pinv(norm_R_p), norm_Y
end

# `_ParametricBatch` admits arrays of any dimension in either form, and `apply_parametric` takes a
# batch only as 3-dimensional arrays, so a batch of any other array is refused by name.
const _TensorPair = NamedTuple{
    (:q, :p), <:Tuple{AbstractArray{<:Any, 3}, AbstractArray{<:Any, 3}}}
_check_batch_form(::AbstractArray{<:Any, 3}, ::AbstractArray{<:Any, 3}) = nothing
_check_batch_form(::_TensorPair, ::_TensorPair) = nothing
function _check_batch_form(input, output)
    throw(ArgumentError(
        "the batch of `SymbolicPullback(arch, ParametricLoss())` is an input and an output of " *
        "2d × 1 × k with k values of μ, or a (q, p) pair of d × 1 × k; it got an input " *
        "$(summary(input)) and an output $(summary(output))."))
end

# The cotangent of the loss at the output: the residual of each half, scaled by that half's inverse
# residual norm and by the norm of the target. Each norm divides each entry on its own, because
# `‖R‖‖Y‖` overflows, and its inverse underflows, where neither norm does.
function (pb::SymbolicPullback{<:ParametricLoss, <:_LayerwiseParameterGradient})(
        ps::NetworkParameters, model, (input, output, μs)::_ParametricBatch)
    _check_batch_form(input, output)
    value, inv_R_q, inv_R_p, norm_Y = _value_and_scales(model, ps, input, output, μs)
    state, target = _sample_columns(input), _sample_columns(output)
    half = size(target, 1) ÷ 2
    TP = parameter_eltype(ps)
    value,
    function (Δ)
        seed = (ŷ, y, _) -> @views vcat(
            (ŷ[1:half, :] .- y[1:half, :]) .* Δ .* inv_R_q ./ norm_Y,
            (ŷ[(half + 1):end, :] .- y[(half + 1):end, :]) .* Δ .* inv_R_p ./ norm_Y)
        gradient = _reseeded(pb.fun.gradient_function, seed)((state, μs), target, ps)
        mapstorage(g -> TP.(g), gradient)
    end
end

# The input of the generated function of one expression, whose columns are the state of a sample
# with that sample's parameters below it.
function _sample_input(input, μs)
    GeometricMachineLearning.concatenate_array_with_parameters(_sample_columns(input), μs)
end

function (pb::SymbolicPullback{<:ParametricLoss, <:_PGHNNParameterGradient})(
        ps::NetworkParameters, model, (input, output, μs)::_ParametricBatch)
    value, inv_R_q, inv_R_p, norm_Y = _value_and_scales(model, ps, input, output, μs)
    TP = parameter_eltype(ps)
    gradient_function = pb.fun.gradient_function
    value,
    Δ -> mapstorage(
        (g_q, g_p) -> TP.((g_q .* inv_R_q .+ g_p .* inv_R_p) .* Δ ./ norm_Y),
        gradient_function(_sample_input(input, μs), _sample_columns(output), ps)...)
end
