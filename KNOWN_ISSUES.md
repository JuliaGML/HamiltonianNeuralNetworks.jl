# Known issues

What is known to be wrong in HamiltonianNeuralNetworks and is not fixed.

## C. Follow-ups and cleanups

### C7 · `SymbolicPullback(::HamiltonianArchitecture)` reaches one private SymbolicNeuralNetworks name.

- **location:** `src/pullbacks/symbolic_hnn_pullback.jl`
- **kind:** upstream
- **found:** 2026-08-16, in GeometricMachineLearning, whose `KNOWN_ISSUES.md` holds it under the same ID
- **evidence:**
  `SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used, because it sizes the symbolic
  target of the loss with `output_dimension(nn.model)`, which for a Hamiltonian network is `1` — the
  scalar Hamiltonian — while `HNNLoss` compares against the Hamiltonian *vector field*, which has the
  dimension of the network's input. The construction therefore reaches
  `SymbolicNeuralNetworks.symbolic_parameter_gradient`, which no exported name replaces:
  `flat_parameter_gradient` is flat, so it wants an `unflatten` per step, and `ParameterGradient` is
  not used. The pullback is the upstream type `SymbolicPullback(loss, fun)`, with a call method of
  this package for `SymbolicPullback{<:HNNLoss, <:_HNNParameterGradient}` that is more specific
  than the upstream call in the pullback and the parameters, and as specific in the data. That
  private name is the one an upstream refactor breaks.

## Found late

### K15 · `HNNLoss` and `SymbolicPullback(arch)` accept `(q, p)` data in their signatures and fail on it inside the generated vector field.

- **location:** `src/loss/hnn_loss.jl:33`
- **kind:** found late
- **found:** GeometricMachineLearning #344, review round 2 (2026-10-14), whose `KNOWN_ISSUES.md` holds it under the same ID; the `HNNLoss` signatures are older than that pull request
- **evidence:**
  `HNNLoss`'s two functors (`src/loss/hnn_loss.jl:31-42`) and the call method of
  `SymbolicPullback(arch)` (`src/pullbacks/symbolic_hnn_pullback.jl:38-40`) type the data as
  `ArrayOrNamedTuple`, which admits a `(q = …, p = …)` `NamedTuple`. The loss hands that data to
  the generated `hvf`, which calls `ndims` on it. With `(q = randn(1, 5), p = randn(1, 5))` as
  input and output, the symbolic pullback, `HNNLoss` and `ZygotePullback(HNNLoss(arch))` all
  raise:

  ```
  MethodError: no method matching ndims(::@NamedTuple{q::Matrix{Float64}, p::Matrix{Float64}})
  ```

  Narrowing the pullback's method alone changes nothing: the data then reaches the upstream
  `SymbolicPullback` call, which also accepts a `NamedTuple` and fails in the same place. The fix
  belongs with the data signatures of the losses as a whole: either each loss accepts only the data
  its vector field can take, or `(q, p)` data is converted before it reaches the field.

## Upstream

### K16 · `train!` on a `Float32` `(q, p)` `ParametricDataLoader` returns a `Float64` history.

- **location:** `test/integration/train_pghnn.jl:82`
- **kind:** upstream
- **found:** 2026-10-27, in GeometricMachineLearning, whose `KNOWN_ISSUES.md` holds it under `K28`
- **evidence:**
  AbstractNeuralNetworks' `_norm` of a `(q, p)` named tuple divides by `√length(dx)`, which is a
  `Float64`, so `ParametricLoss` returns a `Float64` on `Float32` `(q, p)` data and the history of a
  training run on such a loader is a `Vector{Float64}`. The stacked form keeps the data's element
  type. A `Float32` network of `GeneralizedHamiltonianArchitecture(4; parameters = (m = 1.0f0,))`,
  one `Float32` batch of three samples in each form, and the same three `(m = …)` parameters:

  ```julia
  data = (q = rand(Float32, 2, 1, 3), p = rand(Float32, 2, 1, 3))
  stacked = rand(Float32, 4, 1, 3)
  μs = [(m = 1.0f0,), (m = 1.5f0,), (m = 2.0f0,)]
  typeof(ParametricLoss()(nn.model, params(nn), data, data, μs))         # Float64
  typeof(ParametricLoss()(nn.model, params(nn), stacked, stacked, μs))   # Float32
  ```

  The training testset of the `(q, p)` form therefore asserts the length and the finiteness of the
  history and the change of every leaf, and not its element type.

## Raised while adding the parametric networks, not addressed

### K20 · `SymbolicPullback(arch, ParametricLoss())` raises a `MethodError` on a batch with one stacked array and one `(q, p)` pair.

- **location:** `src/pullbacks/symbolic_hnn_pullback.jl:199`
- **kind:** defect
- **found:** #5, review of the check of the batch form
- **evidence:**
  `_ParametricBatch` is a union of two tuple types: an array input with an array output, and a
  `(q, p)` pair input with a `(q, p)` pair output. Neither admits a stacked input with a pair output
  or the reverse, so a call with such a batch matches no method and raises a `MethodError`, not the
  `ArgumentError` that `_check_batch_form` gives a batch of either form whose arrays are not
  3-dimensional with a middle size of 1. The review of #5 ran the call on that pull request's head,
  whose `_ParametricBatch` and call signatures are those of `main`: the forced pullback on a stacked
  input with a pair output and on the reverse, and the layerwise pullback on a stacked input with a
  pair output, each raise `MethodError: no method matching (::SymbolicPullback{ParametricLoss, …})`.
  A `ParametricDataLoader` gives one form for both arrays, so only a hand-built batch reaches it.
