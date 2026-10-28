# [Generalized Hamiltonian Neural Network](@id ghnn_architecture)

A generalized Hamiltonian neural network (GHNN) [horn2025generalized](@cite) learns a Hamiltonian
system as a composition of symplectic-Euler steps rather than as the symplectic gradient of a
scalar network. One step splits the state into ``q`` and ``p`` and changes one of the two by the
gradient of an energy that depends on the other:

```math
\begin{pmatrix} q \\ p \end{pmatrix} \mapsto
\begin{pmatrix} q + \nabla_p T(p) \\ p \end{pmatrix}
\quad\text{or}\quad
\begin{pmatrix} q \\ p \end{pmatrix} \mapsto
\begin{pmatrix} q \\ p - \nabla_q V(q) \end{pmatrix},
```

where ``T`` is a learned kinetic and ``V`` a learned potential energy. A chain of ``n`` such pairs is
a symplectic map from the state to the state one time step later, and it is the map that
[`GeneralizedHamiltonianArchitecture`](@ref) learns.

## Parameter dependence

Each energy network takes the parameters ``\mu`` of the *system* beside the state it differentiates,
so one network covers a range of problems instead of one problem instance. The forward pass is
[`apply_parametric`](@extref GeometricMachineLearning.apply_parametric), reached as the functor
`nn(x, μ)` of the `NeuralNetwork` built around the architecture, and a network trains on a
[`ParametricDataLoader`](@extref GeometricMachineLearning.ParametricDataLoader) with
[`ParametricLoss`](@extref GeometricMachineLearning.ParametricLoss) — the default loss `train!` takes
for that loader.

```julia
using HamiltonianNeuralNetworks
using GeometricMachineLearning: NeuralNetwork, train!, TrainingOptimizer, Adam, Batch, ParametricDataLoader

μ = (m = 1.0, ω = 2.0)
nn = NeuralNetwork(GeneralizedHamiltonianArchitecture(2; parameters = μ))
nn([0.1, 0.2], μ)
```

The parameters an architecture holds are used for their *shape*: they fix the length of the flat
vector that enters an energy network, and the values arrive with each call, one ``\mu`` per sample.

## Forcing and dissipation

[`ForcedGeneralizedHamiltonianArchitecture`](@ref) puts a
[`ForcingLayer`](@extref GeometricMachineLearning.ForcingLayer) between the two steps of each
integrator: a forcing enters the ``\dot{p}`` equation, so it precedes the step that changes ``p``.
Its `forcing_type` names what the forcing reads — `:Q`, `:P` or `:QP` — as in
[`ForcedSympNet`](@extref GeometricMachineLearning.ForcedSympNet).

## The symbolic pullback

[`SymbolicPullback`](@ref SymbolicPullback(::HamiltonianArchitecture)) is the pullback of
[`HNNLoss`](@ref) and takes a scalar Hamiltonian. A generalized Hamiltonian network is a map rather
than a scalar Hamiltonian, so its pullback is
[`SymbolicPullback(arch, ParametricLoss())`](@ref SymbolicPullback(::Union{HamiltonianNeuralNetworks.GeneralizedHamiltonianArchitecture, HamiltonianNeuralNetworks.ForcedGeneralizedHamiltonianArchitecture}, ::HamiltonianNeuralNetworks.ParametricLoss)),
which generates the gradient of the parametric loss, and which `train!` takes as its `loss`.

For a [`GeneralizedHamiltonianArchitecture`](@ref) the pullback is composed layer by layer, by the
layerwise sweep of `SymbolicNeuralNetworks`: fresh symbolic variables between two layers rather
than one layer's expression inside the next. A
[`SymplecticEuler`](@ref HamiltonianNeuralNetworks.SymplecticEuler) step declares what it carries at
that seam: the parameters of the system travel beside the state, flat, one column per sample, and
are put back into their own shape there. Each step is then generated code of its own, and a chain of
any number of integrators builds.

A [`ForcedGeneralizedHamiltonianArchitecture`](@ref) holds a
[`ForcingLayer`](@extref GeometricMachineLearning.ForcingLayer) between two steps, which declares no
such seam, so its pullback is one expression for the whole chain. Each step inlines the generated
gradient of its energy network, and such an expression then holds one copy of it per integrator,
each inside the next, so a forced chain of more than one integrator is refused. `train!` takes a
`ZygotePullback` of the same loss by default, which has no such limit.

## Library Functions

```@docs
HamiltonianNeuralNetworks.GeneralizedHamiltonianArchitecture
HamiltonianNeuralNetworks.ForcedGeneralizedHamiltonianArchitecture
HamiltonianNeuralNetworks.SymbolicEnergy
HamiltonianNeuralNetworks.SymbolicPotentialEnergy
HamiltonianNeuralNetworks.SymbolicKineticEnergy
HamiltonianNeuralNetworks.build_gradient
HamiltonianNeuralNetworks.SymplecticEuler
HamiltonianNeuralNetworks.SymplecticEulerA
HamiltonianNeuralNetworks.SymplecticEulerB
HamiltonianNeuralNetworks.SymbolicPullback(::Union{HamiltonianNeuralNetworks.GeneralizedHamiltonianArchitecture, HamiltonianNeuralNetworks.ForcedGeneralizedHamiltonianArchitecture}, ::HamiltonianNeuralNetworks.ParametricLoss)
```
