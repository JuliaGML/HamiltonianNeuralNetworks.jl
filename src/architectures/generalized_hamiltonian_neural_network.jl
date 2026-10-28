const GHNN_n_integrators_default = 1
const GHNN_n_forcing_layers_default = 2
const GHNN_forcing_type_default = :P

# A symplectic-Euler step splits the state into halves of `dim ÷ 2` rows, which an odd `dim` has no
# split for. The refusal is at construction rather than at the first call.
function _check_dimension(dim::Integer)
    iseven(dim) || throw(ArgumentError(
        "the phase-space dimension of a generalized Hamiltonian network is even, because a " *
        "symplectic-Euler step splits it in halves; it is $(dim)."))
    dim
end

# The number of integrators and the number of hidden layers have a lower bound of their own: a chain
# of no integrator is the identity, and a network has no negative layer.
function _check_n_integrators(n_integrators::Integer)
    n_integrators < 1 && throw(ArgumentError(
        "a generalized Hamiltonian network takes at least one integrator; `n_integrators` is " *
        "$(n_integrators)."))
    n_integrators
end

function _check_nhidden(nhidden::Integer)
    nhidden < 0 && throw(ArgumentError(
        "the number of hidden layers of an energy network is not negative; `nhidden` is " *
        "$(nhidden)."))
    nhidden
end

# A layer of width zero has no output, so an energy network or a forcing network of that width has
# nothing to differentiate.
function _check_width(width::Integer)
    width < 1 && throw(ArgumentError(
        "the width of the layers of a generalized Hamiltonian network is at least one; `width` is " *
        "$(width)."))
    width
end

# The layout of a system without parameters, `NullParameters()` and `NamedTuple()` alike.
const _PARAMETER_FREE_LAYOUT = last(
    GeometricMachineLearning._flatten_system_parameters(NullParameters()))

@doc raw"""
    SymbolicEnergy

The learned energy of one half of a symplectic-Euler step: the shape of the energy network over the
state it differentiates and the parameters of the system, from which `Chain` builds the network and
[`build_gradient`](@ref) the gradient that the step calls.

`parameter_length` and `parameter_layout` are the shape of those parameters: the length of their flat
vector, which is what the first layer of the network adds to its input, and the layout that puts such
a vector back into the shape of `μ`. The names are `ParametricResNetLayer`'s, which carries the same
pair for the same reasons.

A system whose parameters flatten to no entries — `NullParameters()`, `NamedTuple()` or a
`NamedTuple` of empty arrays — has the layout of `NullParameters()`, which is how a
[`SymplecticEuler`](@ref) step built from it knows that it carries nothing.

See [`SymbolicPotentialEnergy`](@ref) and [`SymbolicKineticEnergy`](@ref).
"""
struct SymbolicEnergy{AT <: Activation, Kinetic, LT}
    dim::Int
    width::Int
    nhidden::Int
    parameter_length::Int
    parameter_layout::LT
    activation::AT
end

function SymbolicEnergy(::Val{Kinetic}, dim::Integer, width::Integer, nhidden::Integer,
        activation, parameters::OptionalParameters) where {Kinetic}
    _check_dimension(dim)
    _check_nhidden(nhidden)
    _activation = Activation(activation)
    flat_parameters, layout = GeometricMachineLearning._flatten_system_parameters(
        parameters)
    # A system whose parameters flatten to no entries has none, and the layout of none: the type of the
    # layout is what tells a step whether it carries the parameters across the seam of the layerwise
    # pullback (`src/pullbacks/layerwise_seam.jl`).
    if isempty(flat_parameters)
        layout = _PARAMETER_FREE_LAYOUT
    end
    SymbolicEnergy{typeof(_activation), Kinetic, typeof(layout)}(
        Int(dim), Int(width), Int(nhidden), length(flat_parameters), layout, _activation)
end

@doc raw"""
    SymbolicPotentialEnergy

The potential energy of a generalized Hamiltonian network, ``V`` in the Hamiltonian
``H = T(p) + V(q)`` of a mechanical system. One [`SymplecticEulerB`](@ref) step differentiates it
with respect to ``q``.

The constructor is `SymbolicPotentialEnergy(dim, width, nhidden, activation; parameters)`, where
`dim` is the full phase-space dimension, which is even: the network takes the `dim ÷ 2` rows of
``q`` and the flattened parameters of the system below them.

Also see [`SymbolicKineticEnergy`](@ref).
"""
const SymbolicPotentialEnergy{AT} = SymbolicEnergy{AT, :potential}

@doc raw"""
    SymbolicKineticEnergy

The kinetic energy of a generalized Hamiltonian network, ``T`` in the Hamiltonian
``H = T(p) + V(q)`` of a mechanical system. One [`SymplecticEulerA`](@ref) step differentiates it
with respect to ``p``.

See [`SymbolicPotentialEnergy`](@ref) for the arguments of the constructor.
"""
const SymbolicKineticEnergy{AT} = SymbolicEnergy{AT, :kinetic}

function SymbolicPotentialEnergy(
        dim::Integer, width::Integer, nhidden::Integer, activation;
        parameters::OptionalParameters = NullParameters())
    SymbolicEnergy(Val(:potential), dim, width, nhidden, activation, parameters)
end

function SymbolicKineticEnergy(dim::Integer, width::Integer, nhidden::Integer, activation;
        parameters::OptionalParameters = NullParameters())
    SymbolicEnergy(Val(:kinetic), dim, width, nhidden, activation, parameters)
end

# The energy network: a first dense layer over the `dim ÷ 2` rows of the state it differentiates and
# the flattened parameters of the system, `nhidden` hidden layers of width `width`, and a linear map
# to the scalar energy. The output layer carries no bias.
function Chain(se::SymbolicEnergy)
    hidden = ntuple(_ -> Dense(se.width, se.width, se.activation), se.nhidden)
    Chain(Dense(se.dim ÷ 2 + se.parameter_length, se.width, se.activation), hidden...,
        Linear(se.width, 1; use_bias = false))
end

@doc raw"""
    build_gradient(se::SymbolicEnergy)

Build the function that evaluates the gradient of `se`'s energy with respect to the state it
differentiates.

The generated function takes the state with the flattened parameters of the system below it — the
input the energy network of `se` takes — and the network parameters, and returns a vector with the
`dim ÷ 2` rows of the gradient. It is built once per layer, so that a step of a layer is one
call of generated code rather than a differentiation of its energy network.

Only the first `dim ÷ 2` entries of the input are differentiated: the entries after them are the
parameters of the system, which are not.

The generated function is built with `inplace = false`, so that a caller can differentiate through
a step of the layer with `Zygote`.
"""
function build_gradient(se::SymbolicEnergy)
    nn = SymbolicNeuralNetwork(Chain(se))
    differentials = SymbolicNeuralNetworks.symbolic_differentials(nn.input)[1:(se.dim ÷ 2)]
    gradient = SymbolicNeuralNetworks.symbolic_derivative(
        only(nn.model(nn.input, params(nn))), differentials)
    SymbolicNeuralNetworks.build_nn_function(
        gradient, params(nn), nn.input; inplace = false)
end

@doc raw"""
    SymplecticEuler

One step of the symplectic Euler integrator of a generalized Hamiltonian network, built from a
[`SymbolicEnergy`](@ref).

A step splits the state into `q` and `p` and changes one of the two by the gradient of the energy
of the other:

```math
\begin{pmatrix} q \\ p \end{pmatrix} \mapsto
\begin{pmatrix} q + \nabla_p T(p, \mu) \\ p \end{pmatrix}
\quad\text{or}\quad
\begin{pmatrix} q \\ p - \nabla_q V(q, \mu) \end{pmatrix},
```

for the [`SymplecticEulerA`](@ref) and [`SymplecticEulerB`](@ref) constructors.

# Implementation

The gradient is the `gradient_function` that [`build_gradient`](@ref) generated for the energy
network, which the layer stores beside the network itself. A step is one call of it, and the
parameters of the network are the parameters of `energy_model`: they nest one level deeper than the
layer's own, as for every layer that holds a model.

It also carries the shape of the parameters of the *system* — their flat length, and the layout that
puts a flat vector of them back into the shape of `μ`. The layerwise `SymbolicPullback` of
SymbolicNeuralNetworks carries those parameters flat across the seam between two layers, and this is
what the layer reassembles them with. The layout is the last type parameter, so that the partial
signatures of [`SymplecticEulerA`](@ref) and [`SymplecticEulerB`](@ref) go on matching with the
layout left free. The layout's type also says
whether the layer carries anything across that seam: a system without parameters has the empty
layout, so the number of arguments the seam hands the generated kernels is fixed by the layer's
type.
"""
struct SymplecticEuler{
    M, N, FT <: Base.Callable, MT <: Chain, type, ReturnParameters, PT} <:
       AbstractExplicitLayer{M, N}
    gradient_function::FT
    energy_model::MT
    parameter_length::Int
    parameter_layout::PT
end

# The parameters of the layer are the parameters of the energy network it holds, and so is their
# length: one energy network per layer, and a `Chain` walks the nesting.
parameterlength(integrator::SymplecticEuler) = parameterlength(integrator.energy_model)

function initialparameters(rng::Random.AbstractRNG,
        init_weight::AbstractNeuralNetworks.Initializer, integrator::SymplecticEuler,
        backend::NeuralNetworkBackend, ::Type{T}) where {T}
    initialparameters(rng, init_weight, integrator.energy_model, backend, T)
end

@doc raw"""
    SymplecticEulerA

One step that changes ``q`` by the gradient of a [`SymbolicKineticEnergy`](@ref); see
[`SymplecticEuler`](@ref).
"""
const SymplecticEulerA{M, N, FT, AT, ReturnParameters} = SymplecticEuler{
    M, N, FT, AT, :A, ReturnParameters}

@doc raw"""
    SymplecticEulerB

One step that changes ``p`` by the gradient of a [`SymbolicPotentialEnergy`](@ref); see
[`SymplecticEuler`](@ref).
"""
const SymplecticEulerB{M, N, FT, AT, ReturnParameters} = SymplecticEuler{
    M, N, FT, AT, :B, ReturnParameters}

function SymplecticEulerA(se::SymbolicKineticEnergy; return_parameters::Bool)
    gradient_function = build_gradient(se)
    energy_model = Chain(se)
    SymplecticEuler{se.dim, se.dim, typeof(gradient_function), typeof(energy_model), :A,
        return_parameters, typeof(se.parameter_layout)}(
        gradient_function, energy_model, se.parameter_length, se.parameter_layout)
end

function SymplecticEulerB(se::SymbolicPotentialEnergy; return_parameters::Bool)
    gradient_function = build_gradient(se)
    energy_model = Chain(se)
    SymplecticEuler{se.dim, se.dim, typeof(gradient_function), typeof(energy_model), :B,
        return_parameters, typeof(se.parameter_layout)}(
        gradient_function, energy_model, se.parameter_length, se.parameter_layout)
end

# The input of the energy network: the half of the state the step differentiates, with the
# flattened parameters of the system below it. `concatenate_array_with_parameters` is GML's, as it is
# for the parametric layers of GML itself, and returns the state unchanged without parameters.
#
# The length check is what keeps a `μ` of the wrong shape from being silently truncated by the
# generated gradient function, which reads the entries it was built for and ignores the rest.
function _energy_input(integrator::SymplecticEuler, state, μ)
    input = GeometricMachineLearning.concatenate_array_with_parameters(state, μ)
    size(input, 1) == input_dimension(integrator.energy_model) || throw(ArgumentError(
        "the energy network of this layer takes $(input_dimension(integrator.energy_model)) " *
        "inputs and this call has $(size(input, 1))."))
    input
end

# The step, on the `(q, p)` pair. Both halves are arrays of any shape the energy network's gradient
# function takes, so that a vector, a matrix of samples and a tensor of samples all keep their shape.
function _step(integrator::SymplecticEuler{M, M, FT, MT, :A}, qp, μ,
        ps) where {M, FT, MT}
    (q = qp.q + integrator.gradient_function(_energy_input(integrator, qp.p, μ), ps),
        p = qp.p)
end

function _step(integrator::SymplecticEuler{M, M, FT, MT, :B}, qp, μ,
        ps) where {M, FT, MT}
    (q = qp.q,
        p = qp.p - integrator.gradient_function(_energy_input(integrator, qp.q, μ), ps))
end

function (integrator::SymplecticEuler{M, M, FT, MT, type, false})(
        qp::NamedTupleOfArrays{<:Any, (:q, :p)}, μ::OptionalParameters,
        ps) where {M, FT, MT, type}
    _step(integrator, qp, μ, ps)
end

function (integrator::SymplecticEuler{M, M, FT, MT, type, true})(
        qp::NamedTupleOfArrays{<:Any, (:q, :p)}, μ::OptionalParameters,
        ps) where {M, FT, MT, type}
    _step(integrator, qp, μ, ps), μ
end

# A stacked state has its `(q, p)` split taken here and is stacked again on the way out. A `Chain`
# hands the layer either the state or the pair of the state and the system parameters.
function (integrator::SymplecticEuler{M, M, FT, MT, type, false})(
        x::AbstractArray, μ::OptionalParameters,
        ps) where {M, FT, MT, type}
    output = _step(integrator, GeometricMachineLearning.assign_q_and_p(x, M ÷ 2), μ, ps)
    vcat(output.q, output.p)
end

function (integrator::SymplecticEuler{M, M, FT, MT, type, true})(
        x::AbstractArray, μ::OptionalParameters,
        ps) where {M, FT, MT, type}
    output = _step(integrator, GeometricMachineLearning.assign_q_and_p(x, M ÷ 2), μ, ps)
    (vcat(output.q, output.p), μ)
end

function (integrator::SymplecticEuler)(
        qp_and_μ::Tuple{<:NamedTupleOfArrays{<:Any, (:q, :p)}, <:OptionalParameters},
        ps)
    integrator(qp_and_μ..., ps)
end

function (integrator::SymplecticEuler)(
        x_and_μ::Tuple{<:AbstractArray, <:OptionalParameters}, ps)
    integrator(x_and_μ..., ps)
end

# Without system parameters the step takes `NullParameters`, which leaves the state as the input of
# the energy network.
function (integrator::SymplecticEuler)(
        qp::NamedTupleOfArrays{<:Any, (:q, :p)}, ps)
    integrator(qp, NullParameters(), ps)
end

function (integrator::SymplecticEuler)(x::AbstractArray, ps)
    integrator(x, NullParameters(), ps)
end

@doc raw"""
    GeneralizedHamiltonianArchitecture <: HamiltonianArchitecture

A realization of generalized Hamiltonian neural networks (GHNNs) as introduced in
[horn2025generalized](@cite).

The architecture is a chain of `n_integrators` pairs of symplectic-Euler steps, `2 n_integrators`
steps in all: each pair differentiates a learned kinetic and a learned potential energy. Its forward
pass takes the parameters of the *system* beside the state:

```julia
nn(x, μ)
```

where `x` is a `(q, p)` pair or a stacked state and `μ` the parameters of the system, as
[`apply_parametric`](@extref GeometricMachineLearning.apply_parametric) takes them. This is what
makes one network cover a range of problems rather than one problem instance.

Also see [`StandardHamiltonianArchitecture`](@ref) and
[`ForcedGeneralizedHamiltonianArchitecture`](@ref).

# Constructor

The constructor takes the following input arguments:
1. `dim`: system dimension, which is even,
2. `width = dim`: width of the hidden layer. By default this is equal to `dim`,
3. `nhidden = 1`: the number of hidden layers of each energy network,
4. `n_integrators = 1`: the number of integrators used in the GHNN,
5. `activation = AbstractNeuralNetworks.TanhActivation()`: the activation function used in the GHNN,
6. `parameters = NullParameters()`: the parameters of the system, which fix the length of the flat
   vector the energy networks take beside their state; their values enter with each call.

# The parameter count

One integrator holds two energy networks, one kinetic and one potential, and neither is shared, so
the network has `2 n_integrators` of them. Each is a first dense layer over `dim ÷ 2 + |\mu|` rows
and `width` columns, `nhidden` hidden layers of width `width`, and a biasless `Linear(width, 1)`, so
the whole network has

```math
2\,n_\mathrm{integrators}\left[(\mathrm{dim}/2 + |\mu| + 1)\mathrm{width} +
\mathrm{nhidden}\,\mathrm{width}(\mathrm{width} + 1) + \mathrm{width}\right]
```

parameters. The thesis counts the first dense layer among the hidden layers, i.e. as `nhidden + 1` of
them.
"""
struct GeneralizedHamiltonianArchitecture{AT, PT <: OptionalParameters} <:
       HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    n_integrators::Int
    parameters::PT
    activation::AT

    function GeneralizedHamiltonianArchitecture(dim::Integer; width::Integer = dim,
            nhidden::Integer = HNN_nhidden_default,
            n_integrators::Integer = GHNN_n_integrators_default, activation = HNN_activation_default,
            parameters::OptionalParameters = NullParameters())
        _check_dimension(dim)
        _check_width(width)
        _check_nhidden(nhidden)
        _check_n_integrators(n_integrators)
        _activation = Activation(activation)
        new{typeof(_activation), typeof(parameters)}(
            Int(dim), Int(width), Int(nhidden), Int(n_integrators), parameters, _activation)
    end
end

function Chain(arch::GeneralizedHamiltonianArchitecture)
    kinetic_energy = SymbolicKineticEnergy(
        arch.dim, arch.width, arch.nhidden, arch.activation;
        parameters = arch.parameters)
    potential_energy = SymbolicPotentialEnergy(arch.dim, arch.width, arch.nhidden,
        arch.activation; parameters = arch.parameters)

    layers = ()
    for n in 1:arch.n_integrators
        layers = (layers..., SymplecticEulerA(kinetic_energy; return_parameters = true))
        layers = (layers...,
            SymplecticEulerB(potential_energy;
                return_parameters = n != arch.n_integrators))
    end

    Chain(layers...)
end

# Without this, `nn(x, μ)` reaches `AbstractNeuralNetworks`' generic two-argument functor, which
# reads the system parameters as the *network* parameters and hands the first layer a `Float64`. The
# method takes exactly the forms `apply_parametric` takes — one `μ`, or any `AbstractVector` of them
# — and raises its `MethodError` for a shape it does not, a matrix with one `μ` among them.
function (nn::NeuralNetwork{<:GeneralizedHamiltonianArchitecture})(
        x, μ::Union{OptionalParameters, AbstractVector})
    apply_parametric(nn.model, x, μ, params(nn))
end
