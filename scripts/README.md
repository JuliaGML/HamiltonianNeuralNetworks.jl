# `scripts/`

Three directories, each with one purpose.

| directory | holds | how CI runs it |
|:--|:--|:--|
| `verification/` | checks that establish a mathematical claim about the package | in full |
| `reproduction/` | the training runs that reproduce a result, with its figure | in smoke mode |
| `utilities/` | what the other two include | not run |

`scripts/runscripts.jl` is the driver, and `.github/workflows/Scripts.yml` is the job that calls
it. Every `.jl` file under `verification/` and `reproduction/` is an entry point and is run, each
in its own process. There is no list of what to run, so a script added to either directory is
gated from the moment it lands — and a file that is *included* rather than run belongs in
`utilities/`, or the driver will try to run it on its own and fail.

```sh
julia --startup-file=no --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=scripts scripts/runscripts.jl verification
julia --startup-file=no --project=scripts scripts/runscripts.jl reproduction
```

## Size constants

A reproduction script trains, and training takes time. Every constant that costs time is written

```julia
include("../utilities/smoke.jl")
const nepochs = smoke_size(2000, 2)
```

`smoke_size` returns its second argument when `HNN_SMOKE` is set in the environment, which is what
the CI job sets. A smoke run establishes that the script still executes end to end, and nothing
about the result.

To reproduce a result, run the script with `HNN_SMOKE` unset. That is the default.

## What the gate does not cover

Nothing. `SKIPPED` in `scripts/runscripts.jl` names the scripts the job does not run, and it is
empty: every entry point under `verification/` and `reproduction/` is executed on each run.
