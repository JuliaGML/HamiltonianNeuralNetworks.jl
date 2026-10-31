# How `SymplecticEuler` meets the seam of the layerwise sweep of `SymbolicNeuralNetworks`.
#
# The sweep composes the pullback of a chain layer by layer, putting fresh symbolic variables between
# two layers rather than inlining one layer's expression into the next, which keeps the symbolic
# material a sum over layers rather than a product. By default that seam is one plain vector — the
# state — so the construction assumes every layer maps an array to an array.
#
# A `SymplecticEuler` built with `return_parameters = true` does not: it threads the parameters of the
# *system* on to the next layer, so it takes and returns the pair `(state, μ)`. The four functions
# below say how it meets the seam, in the interface `SymbolicNeuralNetworks.seam_interface`
# describes. `SymbolicPullback(arch, ParametricLoss())` hands the sweep a `2d × k` matrix of samples
# and the vector of their `k` values of `μ`.

# Whether the layer carries the parameters of the system across the seam is a property of its type,
# because the sweep splats what `seam_arguments` returns into the generated kernels and that call
# infers only if the length of the tuple is known from the type. The layout of the parameters is the
# layer's last type parameter, and a system without them — `NullParameters()`, `NamedTuple()`, or
# parameters that flatten to no entries — has the empty layout, so a test of the layout's type is
# folded away by the compiler.
const _ParameterFreeLayout = typeof(_PARAMETER_FREE_LAYOUT)

function _carries_parameters(integrator::SymplecticEuler)
    !(integrator.parameter_layout isa _ParameterFreeLayout)
end

# A system with no parameters carries nothing, and says so with an empty tuple rather than an empty
# array: there is nothing to hand the generated kernels, and the seam is then the plain vector it is
# for every other layer. The layer is still applied without them, which its own two-argument functor
# supplies as `NullParameters()`.
function SymbolicNeuralNetworks.carried_variables(integrator::SymplecticEuler)
    _carries_parameters(integrator) ?
    (Symbolics.variables(:π, 1:integrator.parameter_length),) : ()
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

# The run-time counterpart of `seam_value`: every data argument of a generated function has one column
# per sample, so the parameters of sample `i` are column `i` of the flat matrix, in the element type
# of the state. `μ` arrives in whatever precision it was written in, and a gradient that kept the
# precision of `μ` would not be the state's.
function SymbolicNeuralNetworks.seam_arguments(
        integrator::SymplecticEuler, (state, μs)::Tuple{<:AbstractMatrix, <:AbstractVector})
    _carries_parameters(integrator) || return (state,)
    flat = [first(GeometricMachineLearning.flatten_system_parameters(eltype(state), μ))
            for μ in μs]
    (state, reduce(hcat, flat))
end
