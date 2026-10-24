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

### Dependencies

* **Julia 1.12 is required.** `[compat]` declares `julia = "1.12"`, so a caller on Julia 1.11
  cannot install this version. The package is carved out of GeometricMachineLearning at that
  floor: the commit that creates it declares the 1.12 bound from the start, so 1.12 is where the
  package begins rather than a floor raised later.
