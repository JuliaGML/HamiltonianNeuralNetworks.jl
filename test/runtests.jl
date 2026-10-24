using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua's package-level checks" include("quality/aqua.jl")
    @safetestset "Hamiltonian Neural Network" include("architectures/hamiltonian_neural_network_tests.jl")
    @safetestset "Symplectic Euler loss" include("loss/training_method_losses.jl")
    @safetestset "SymbolicPullback of HNNLoss is the gradient of the batched loss" include("pullbacks/symbolic_hnn_pullback.jl")
    @safetestset "A loss takes a NetworkParameters" include("loss/losses.jl")
    @safetestset "train! with a Hamiltonian network" include("integration/train_with_symbolic_pullback.jl")
end
if "slow" in GROUPS
    @safetestset "train! trains every architecture of the manual" include("integration/train_manual_architectures.jl")
end
