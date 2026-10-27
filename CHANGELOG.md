# Release Notes

All notable changes to HamiltonianNeuralNetworks.jl.

This package is pre-1.0, so every minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions.


## [Unreleased] — targeting 0.1.0

### New Features

* **The package, moved out of GeometricMachineLearning.** The Hamiltonian neural network of
  GeometricMachineLearning.jl moves here with its history, from GML commit `e759f206` on its
  `main` branch: `HamiltonianArchitecture`, `StandardHamiltonianArchitecture`,
  `GeneralizedHamiltonianArchitecture` (a constructor that raises an error, as in GML), the losses
  `HNNLoss` and `SymplecticEulerLoss`, and the `SymbolicPullback(::HamiltonianArchitecture)`
  method, with `SymbolicPullback` re-exported from SymbolicNeuralNetworks. `git log --follow` of
  each moved file reaches its GML history.
* **Training goes through GeometricMachineLearning by name.** The package does not depend on
  GML, so a user adds it to their own environment, from GML's `main` branch until 0.9.0 is
  registered. The tests, the manual and the scripts load `DataLoader`, `Batch`, `train!` and the
  optimisers with `using GeometricMachineLearning: …`, never with a bare `using`, because GML
  exports the same names for its own copy of the architecture until it removes it. Until GML
  0.9.0 is registered, the test, docs and scripts environments take GML from its `main` branch
  through `[sources]`.
* **Tests in both precisions.** The tests of the architecture and of `SymplecticEulerLoss` run in
  `Float32` and `Float64` and assert the element type of the loss, the gradient and the training
  history. The tests of the mixed GML files keep their Hamiltonian lines: the
  `SymplecticEulerLoss` lines of `test/loss/training_method_losses.jl`, the Hamiltonian entries
  of `test/loss/losses.jl`, the `SymbolicPullback` training test and the Hamiltonian line of the
  default-loss test of GML's `test/data_loader/train.jl`, which moves to
  `test/integration/train_with_symbolic_pullback.jl`, and the Hamiltonian entry of
  `test/integration/train_manual_architectures.jl`.
* **The manual and the scripts.** The architecture page and the tutorial move with their
  bibliography entries, and link into GML's manual through DocumenterInterLinks. The pendulum
  reproduction scripts and the `SymplecticEulerLoss` verification script move with their
  utilities; `smoke_size` reads `HNN_SMOKE`, and `scripts/runscripts.jl` runs them.

* **The generalized and forced Hamiltonian networks, with the parameters of the system.**
  `GeneralizedHamiltonianArchitecture` is implemented rather than a constructor that raises: it is a
  chain of `n_integrators` pairs of symplectic-Euler steps, each pair differentiating a learned
  kinetic and a learned potential energy, and its forward pass takes the parameters of the *system*
  beside the state — `nn(x, μ)`, or `apply_parametric` — so one network covers a range of problems
  rather than one instance. `ForcedGeneralizedHamiltonianArchitecture` puts a `ForcingLayer` between
  the two steps of each integrator and is exported beside the other two architectures.
  `SymbolicEnergy` with its kinetic and potential constructors, `SymplecticEuler` with its `A` and
  `B` steps and `build_gradient` stay internal. Ported from GeometricMachineLearning.jl#207.
* **`SymbolicPullback(arch, ParametricLoss())`.** The gradient of the parametric loss for a
  generalized or forced network, generated in closed form: the per-sample gradients of the surrogate
  `½‖rᵢ‖²`, summed over the batch and scaled by the residual and target norms that loss gives that
  batch form, in both precisions. Its value and gradient are those of `ParametricLoss` on the batch,
  which a `ParametricDataLoader` draws as a `2d × 1 × k` tensor or as a `(q, p)` pair of `d × 1 × k`
  ones; the loss weighs the two halves of a pair by their own norms, and the pullback does the same.
  A chain of more than one integrator is refused with an `ArgumentError`, because each step of the
  chain inlines the generated gradient of its energy network and one expression for the whole chain
  grows with every integrator; `train!` takes the `ZygotePullback` of the same loss.
  `SymbolicPullback(arch::HamiltonianArchitecture)` is the pullback of `HNNLoss` and refuses a
  generalized architecture, which is a map rather than a scalar Hamiltonian.
* **The forced generalized Hamiltonian network's reproduction script.** It trains the three forcing
  types of the sinusoidally forced harmonic oscillator — where the time is the parameter of the
  system — and integrates a trajectory with the learned network; `HNN_SMOKE` runs it at a size that
  exercises the same code path in seconds.

### Dependencies

* **GeometricMachineLearning is a dependency.** The forced architecture is built from GML's
  `ForcingLayer`, and the parametric network trains through GML's `ParametricLoss` and
  `ParametricDataLoader`, so the package depends on GML rather than asking a caller to supply it.
  It loads Zygote with it. Until GML 0.9.0 is registered, `[sources]` takes GML from its `main`
  branch.
* **GeometricBase supplies the parameters of the system.** `NullParameters` and
  `OptionalParameters` are what an architecture's `parameters` field and every parametric call take;
  GeometricBase owns both names, and reaching them from the package that defines them keeps
  ExplicitImports' owner check green.

* **Julia 1.12 is required.** `[compat]` declares `julia = "1.12"`, so a caller on Julia 1.11
  cannot install this version. The package is carved out of GeometricMachineLearning at that
  floor: the commit that creates it declares the 1.12 bound from the start, so 1.12 is where the
  package begins rather than a floor raised later.
