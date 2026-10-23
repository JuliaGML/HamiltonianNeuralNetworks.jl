# HamiltonianNeuralNetworks

`HamiltonianNeuralNetworks` learns Hamiltonian vector fields from data. Its network approximates
the Hamiltonian, and the vector field is the symplectic gradient of the network
[greydanus2019hamiltonian](@cite).

The package provides

- the [architecture](@ref hnn_architecture) [`StandardHamiltonianArchitecture`](@ref),
- the losses [`HNNLoss`](@ref), for vector field data, and [`SymplecticEulerLoss`](@ref), for
  pairs of states one timestep apart,
- the [`SymbolicPullback`](@ref SymbolicPullback(::HamiltonianArchitecture)) of `HNNLoss`, a
  gradient that SymbolicNeuralNetworks generates in closed form.

The networks train with the `DataLoader`, `Batch`, `train!` and the optimisers of
[GeometricMachineLearning](https://github.com/JuliaGNI/GeometricMachineLearning.jl); the
[tutorial](@ref hnn_tutorial) shows the whole workflow.

```@docs
HamiltonianNeuralNetworks
```
