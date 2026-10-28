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

### K22 · `test/pullbacks/symbolic_pghnn_pullback.jl` takes 62–64 s, above the 60 s that a file of the `core` group may take.

- **location:** `test/pullbacks/symbolic_pghnn_pullback.jl`
- **kind:** found late
- **found:** 2026-10-28, in the review of the layerwise seam, round 1
- **evidence:**
  `run-tests.jl` reports a `core` file whose cold time is above 60 s, and both `core` runs of the
  review report this one: `D6: test/pullbacks/symbolic_pghnn_pullback.jl runs 62.3 s cold, above the
  60 s of a core file` in the critic's, 64.0 s in the build's. A `Pkg.test()` at a load average of
  39.7 spends 64.4 s of testset time on the file. The time follows the load rather than the code:
  the file holds the same statements on `origin/main` and on the branch, which rewords one comment in
  it, and run alone from a cold process it takes `selected |471 471 1m30.8s` on a tree of
  `origin/main` at a load average of 24.6 and `selected |471 471 1m27.3s` on the branch at 26.0 — the
  tree with the seam's two extra fields and type parameter is the faster of the two. Moving the file
  to the `slow` group is what closes it.

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

### K20 · The layerwise `SymbolicPullback` of `ParametricLoss` gives a gradient, but calling the pullback raises a `MethodError`, so `SymbolicPullback(arch, ParametricLoss())` keeps its refusal.

- **location:** `src/pullbacks/layerwise_seam.jl`
- **kind:** upstream
- **found:** 2026-10-27, in the layerwise seam; GeometricMachineLearning #245's remaining half
- **evidence:**
  The layerwise construction of SymbolicNeuralNetworks takes the gradient as `pullback.fun(input,
  output, ps)(Δ)`, and that works for a parametrized generalized Hamiltonian network: on
  `GeneralizedHamiltonianArchitecture(4; width = 2, nhidden = 1, parameters = (m = 1.0, ω = π/2))`
  it returns a `Vector{Float64}` of the parameters' shape, and `test/pullbacks/layerwise_seam.jl`
  checks it against `Zygote`. The pullback as a whole does not: its two functors evaluate the value
  as `pullback.loss(model, ps, input, output)` — four arguments — and `ParametricLoss` takes five,
  with the system parameters last. Calling it raises, with the parameter types elided:

  ```
  MethodError: no method matching (::ParametricLoss)(::Chain{…}, ::NetworkParameters{…},
      ::Tuple{Vector{Float64}, @NamedTuple{m::Float64, ω::Float64}}, ::Vector{Float64})
  ```

  So `train!` cannot take this pullback, and `SymbolicPullback(arch, ParametricLoss())` keeps
  refusing a chain of more than one integrator. `ZygotePullback(ParametricLoss())` is what trains
  such a network. A four-argument form of `ParametricLoss`, or a layerwise construction whose call
  takes the system parameters, is what closes this.

### K21 · A `(q, p)` pair reaches `batched`, which has no `NamedTuple` method.

- **location:** `src/pullbacks/layerwise_seam.jl`
- **kind:** upstream
- **found:** 2026-10-27, in the layerwise seam
- **evidence:**
  `SymbolicNeuralNetworks`' layerwise sweep lays its input out with `batched`, which has methods for
  an `AbstractVector`, an `AbstractMatrix`, any other `AbstractArray` and a `Tuple`, and none for a
  `NamedTuple`. The `seam_interface` of that package names this as the batch that has to be laid
  flat, and a `(q, p)` pair — the input form the parametric architectures of this package take — is
  not one of them. With `pair = (q = rand(2, 1), p = rand(2, 1))` and one `μ`:

  ```
  MethodError: no method matching batched(::@NamedTuple{q::Matrix{Float64}, p::Matrix{Float64}})
  ```

  `SymbolicNeuralNetworks.seam_arguments` is reached after that call, so the pair never gets to the
  seam this package declares for it. A `batched` method that stacks the two halves closes it.

## Raised while adding the parametric networks, not addressed

### K17 · The `GeometricMachineLearning` compat bound admits the registered 0.8.0, which has none of the parametric names the package imports.

- **location:** `Project.toml:22`
- **kind:** defect
- **found:** 2026-10-27, in the pre-pull-request verification of the parametric networks
- **evidence:**
  The package imports `ForcingLayer`, `ParametricLoss` and `apply_parametric` from
  GeometricMachineLearning (`src/HamiltonianNeuralNetworks.jl:19`), and reaches
  `_flatten_system_parameters`, `concatenate_array_with_parameters`, `assign_q_and_p` and
  `_check_forcing_type` through the module's name. The registered 0.8.0 has six of those seven names
  nowhere: its tree is `7616b841d9f7fca26df41a83ea8c2496ea31d25a`, which is this checkout's
  `v0.8.0`, `git grep -l` finds `ForcingLayer`, `ParametricLoss`, `apply_parametric`,
  `_flatten_system_parameters`, `concatenate_array_with_parameters` and `_check_forcing_type` at no
  file under `v0.8.0:src/`, and `assign_q_and_p` is the one it has. A
  resolve that chose that release therefore raises an `UndefVarError` on load. It cannot be seen
  here, because `[sources]` takes GML from commit `2604f115` of its `main` branch, which declares
  `version = "0.8.0"` and so satisfies the bound; a bound of `"0.9"` fails to resolve against it.
  The bound therefore stays `"0.8"` until GML declares 0.9. It becomes `"0.9"` in the change that
  removes `[sources]` from this file, once GML 0.9 is registered: the parametric API is GML 0.9's,
  and a release against 0.8.0 has no such names to import.

### K19 · The Downgrade job fails to build while `Project.toml` takes GeometricMachineLearning from a URL source.

- **location:** `Project.toml:17`
- **kind:** defect
- **found:** #3
- **evidence:**
  `julia-downgrade-compat` logs `Found source package: GeometricMachineLearning (url=…)` and leaves
  GML out of the dependencies it pins, and `julia-buildpkg` then stops with
  `LoadError: could not find manifest entry for package with uuid 194d25b2-d3f5-49f0-af24-c124f4aa80cc`,
  which is GML's. The job builds nothing, so it tests no lower bound. It is advisory
  (`.github/workflows/CI.yml:25-27`), and it stays red until the change of K17 removes `[sources]`.
