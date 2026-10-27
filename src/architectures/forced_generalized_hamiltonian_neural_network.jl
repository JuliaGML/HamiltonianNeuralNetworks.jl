@doc raw"""
    ForcedGeneralizedHamiltonianArchitecture <: HamiltonianArchitecture

A [`GeneralizedHamiltonianArchitecture`](@ref) with a non-conservative term — a forcing or a
dissipation — in each of its integrators.

Each integrator is a [`SymplecticEulerA`](@ref) step, a
[`ForcingLayer`](@extref GeometricMachineLearning.ForcingLayer), and a [`SymplecticEulerB`](@ref)
step, in that order: the forcing acts on ``\dot{p}``, so it precedes the step that changes ``p``.
Also compare [`ForcedSympNet`](@extref GeometricMachineLearning.ForcedSympNet).

# Constructor

The constructor takes the arguments of [`GeneralizedHamiltonianArchitecture`](@ref) and
1. `n_forcing_layers = 2`: the number of hidden layers of the network of each forcing layer,
2. `forcing_type = :P`: what the forcing reads, `:Q`, `:P` or `:QP`; see
   [`ForcingLayer`](@extref GeometricMachineLearning.ForcingLayer).
"""
struct ForcedGeneralizedHamiltonianArchitecture{FT, AT, PT <: OptionalParameters} <:
       HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    n_forcing_layers::Int
    n_integrators::Int
    parameters::PT
    activation::AT

    function ForcedGeneralizedHamiltonianArchitecture(dim::Integer; width::Integer = dim,
            nhidden::Integer = HNN_nhidden_default,
            n_forcing_layers::Integer = GHNN_n_forcing_layers_default,
            n_integrators::Integer = GHNN_n_integrators_default, activation = HNN_activation_default,
            parameters::OptionalParameters = NullParameters(),
            forcing_type::Symbol = GHNN_forcing_type_default)
        _check_dimension(dim)
        _check_width(width)
        _check_nhidden(nhidden)
        _check_n_integrators(n_integrators)
        n_forcing_layers < 0 && throw(ArgumentError(
            "the number of hidden layers of a forcing layer is not negative; " *
            "`n_forcing_layers` is $(n_forcing_layers)."))
        # GML's own guard, so that the refused set is the set `ForcingLayer` reads rather than a copy
        # of it. The architecture refuses a mistyped `forcing_type` at construction, before a `Chain`
        # exists.
        GeometricMachineLearning._check_forcing_type(forcing_type)
        _activation = Activation(activation)
        new{forcing_type, typeof(_activation), typeof(parameters)}(
            Int(dim), Int(width), Int(nhidden), Int(n_forcing_layers), Int(n_integrators),
            parameters, _activation)
    end
end

function Chain(arch::ForcedGeneralizedHamiltonianArchitecture{FT}) where {FT}
    kinetic_energy = SymbolicKineticEnergy(
        arch.dim, arch.width, arch.nhidden, arch.activation;
        parameters = arch.parameters)
    potential_energy = SymbolicPotentialEnergy(arch.dim, arch.width, arch.nhidden,
        arch.activation; parameters = arch.parameters)

    layers = ()
    for n in 1:arch.n_integrators
        layers = (layers..., SymplecticEulerA(kinetic_energy; return_parameters = true))
        layers = (layers...,
            ForcingLayer(arch.dim, arch.width, arch.n_forcing_layers, arch.activation;
                parameters = arch.parameters, return_parameters = true, type = FT))
        # The last step of the chain returns the state alone, so that the chain's output is a state.
        layers = (layers...,
            SymplecticEulerB(potential_energy;
                return_parameters = n != arch.n_integrators))
    end

    Chain(layers...)
end

# `ForcedGeneralizedHamiltonianArchitecture` and `GeneralizedHamiltonianArchitecture` are siblings
# under `HamiltonianArchitecture`, so the parameter-dependent functor of the latter does not cover
# this one, and without a method of its own `nn(x, μ)` falls through to
# `AbstractNeuralNetworks`' generic two-argument functor, which reads the system parameters as the
# network parameters.
function (nn::NeuralNetwork{<:ForcedGeneralizedHamiltonianArchitecture})(
        x, μ::Union{OptionalParameters, AbstractVector})
    apply_parametric(nn.model, x, μ, params(nn))
end
