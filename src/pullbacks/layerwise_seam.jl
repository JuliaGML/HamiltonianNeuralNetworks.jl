# How `SymplecticEuler` meets the seam of the layerwise `SymbolicPullback`.
#
# `SymbolicNeuralNetworks` composes the pullback of a chain layer by layer, putting fresh symbolic
# variables between two layers rather than inlining one layer's expression into the next, which keeps
# the symbolic material a sum over layers rather than a product. By default that seam is one plain
# vector — the state — so the construction assumes every layer maps an array to an array.
#
# A `SymplecticEuler` built with `return_parameters = true` does not: it threads the parameters of the
# *system* on to the next layer, so it takes and returns the pair `(state, μ)`. The four methods below
# say how it meets the seam, in the interface `SymbolicNeuralNetworks.seam_interface` describes; a
# layer that declares none of them is declined, and the monolithic construction takes over, which for
# a system with parameters does not build at all.

# A system with no parameters carries nothing, and says so with an empty tuple rather than an empty
# array: there is nothing to hand the generated kernels, and the seam is then the plain vector it is
# for every other layer. The layer is still applied without them, which its own two-argument functor
# supplies as `NullParameters()`.
function SymbolicNeuralNetworks.carried_variables(integrator::SymplecticEuler)
    iszero(integrator.parameter_length) ? () :
    (Symbolics.variables(:π, 1:integrator.parameter_length),)
end

# The generated kernels take the parameters of the system flat, which is how the forward pass feeds
# them to an energy network as well; the layer itself wants them in the shape it was built with, so
# the seam puts them back through its layout.
function SymbolicNeuralNetworks.seam_value(integrator::SymplecticEuler, sqp, sparameters)
    (sqp, unflatten(integrator.parameter_layout, sparameters))
end

# `return_parameters = true` returns the state *and* the parameters of the system, and the
# sensitivity of the loss pairs with the state. With `false` the output is the state alone, which is
# what the default of `state_expressions` reads.
function SymbolicNeuralNetworks.state_expressions(
        ::SymplecticEuler{M, N, FT, AT, kind, true}, y) where {M, N, FT, AT, kind}
    SymbolicNeuralNetworks.scalar_expressions(first(y))
end

# The run-time counterpart of `seam_value`. Every data argument of a generated function has the
# state's rank and batch size, so one set of parameters for the whole call is repeated under each
# column of a matrix, as the forward pass does with them.
function SymbolicNeuralNetworks.seam_arguments(integrator::SymplecticEuler, x::Tuple)
    state = _seam_state(first(x))
    iszero(integrator.parameter_length) ? (state,) :
    (state, _seam_parameters(last(x), state))
end

# The state alone is a seam only for a system without parameters: the kernels of a layer that
# carries them take the state and the flat parameters, and one data argument would be a `MethodError`
# inside generated code, where the caller cannot see what is missing. Through the sweep the length
# check of `_energy_input` raises first; this is the guard for a direct call.
function SymbolicNeuralNetworks.seam_arguments(integrator::SymplecticEuler,
        qp::Union{AbstractArray, NamedTupleOfArrays{<:Any, (:q, :p)}})
    iszero(integrator.parameter_length) || throw(ArgumentError(
        "the system of this `SymplecticEuler` has $(integrator.parameter_length) parameter(s), so " *
        "the pullback has to be given `(state, system parameters)` rather than the state alone."))
    (_seam_state(qp),)
end

# The seam holds the `dim` components of the state as one array, which is how the layer's own array
# methods take the state too; a `(q, p)` pair is stacked as `assign_q_and_p` takes it apart again.
_seam_state(qp::AbstractArray) = qp
_seam_state(qp::NamedTupleOfArrays{<:Any, (:q, :p)}) = vcat(qp.q, qp.p)

# The parameters of the system flattened in the element type of the state: `μ` arrives in whatever
# precision it was written in, and a gradient that kept the precision of `μ` would not be the
# state's.
function _seam_parameters(parameters::OptionalParameters, state::AbstractVector)
    first(GeometricMachineLearning._flatten_system_parameters(eltype(state), parameters))
end

function _seam_parameters(parameters::OptionalParameters, state::AbstractMatrix)
    repeat(
        first(GeometricMachineLearning._flatten_system_parameters(eltype(state), parameters)),
        1, size(state, 2))
end

# `ParametricLoss` applies the loss in a five-argument form, so the layerwise construction cannot
# guess an expression of the prediction and the target from it and would decline — falling back to
# the monolithic construction, which for a parameter-dependent architecture does not build. The
# declaration restates `_compute_loss`, which is the relation the loss's own functor makes, so the two
# cannot drift.
function SymbolicNeuralNetworks.loss_expression(::ParametricLoss, ŷ, y)
    AbstractNeuralNetworks._compute_loss(ŷ, y)
end
