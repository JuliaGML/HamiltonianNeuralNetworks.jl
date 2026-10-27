using Aqua
using ExplicitImports
using GeometricMachineLearning: ParametricLoss
using HamiltonianNeuralNetworks
using Test

# Aqua's package-level checks, all eight, with an enclosing testset: `Aqua.test_all` adds none, so
# without it the first failing check would end the file and hide the others.
#
# `ParametricLoss` is GeometricMachineLearning's type, and this package declares one method of
# SymbolicNeuralNetworks' `loss_expression` on it — the layerwise `SymbolicPullback` cannot guess the
# expression of a loss with five arguments, and the declaration belongs beside the loss it restates.
# Aqua counts a method as piracy when neither the function it extends nor a type in its signature is
# this package's, so that one type is named as this package's own; every other method is still
# checked.
@testset "Aqua" begin
    Aqua.test_all(HamiltonianNeuralNetworks; piracies = (treat_as_own = [ParametricLoss],))
end

# A `using` or an `import` that nothing needs keeps a dependency reachable from the module file, so
# Aqua's `stale_deps` above sees a package in use that nothing uses. The two checks of public names
# run below with the names this package reaches in its dependencies that they do not declare public.
test_explicit_imports(HamiltonianNeuralNetworks;
    all_explicit_imports_are_public = false,
    all_qualified_accesses_are_public = false)

const NON_PUBLIC_QUALIFIED = (
    :TanhActivation, :dim, :Initializer,          # AbstractNeuralNetworks
    :_compute_loss,                               # AbstractNeuralNetworks
    :Callable,                                    # Base
    :_flatten_system_parameters,                  # GeometricMachineLearning
    :concatenate_array_with_parameters,           # GeometricMachineLearning
    :assign_q_and_p,                              # GeometricMachineLearning
    :_check_forcing_type,                         # GeometricMachineLearning
    :Jacobian, :symbolic_parameter_gradient,      # SymbolicNeuralNetworks
    :symbolic_differentials, :symbolic_derivative, # SymbolicNeuralNetworks
    :carried_variables, :seam_value,              # SymbolicNeuralNetworks
    :state_expressions, :seam_arguments,          # SymbolicNeuralNetworks
    :scalar_expressions, :loss_expression,        # SymbolicNeuralNetworks
    :variables)                                   # Symbolics
const NON_PUBLIC_IMPORTS = (
    :AbstractExplicitLayer, :Architecture,       # AbstractNeuralNetworks
    :derivative)                                 # SymbolicNeuralNetworks

@testset "every other name reached in a dependency is public there" begin
    @test check_all_qualified_accesses_are_public(HamiltonianNeuralNetworks;
        ignore = NON_PUBLIC_QUALIFIED) === nothing
    @test check_all_explicit_imports_are_public(HamiltonianNeuralNetworks;
        ignore = NON_PUBLIC_IMPORTS) === nothing
end
