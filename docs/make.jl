using HamiltonianNeuralNetworks
using Documenter
using DocumenterCitations
using DocumenterInterLinks

# The pages link into the manual of GeometricMachineLearning, which holds the data loader, the
# training loop and the background on symplectic systems. Its `latest/` inventory is the one of its
# `main` branch, which the docs environment takes GeometricMachineLearning from.
links = InterLinks(
    "GeometricMachineLearning" => "https://juliagni.github.io/GeometricMachineLearning.jl/latest/"
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "HamiltonianNeuralNetworks.bib"))

makedocs(;
    plugins = [bib, links],
    modules = [HamiltonianNeuralNetworks],
    authors = "Michael Kraus, Benedikt Brantner",
    repo = "https://github.com/JuliaGML/HamiltonianNeuralNetworks.jl/blob/{commit}{path}#L{line}",
    sitename = "HamiltonianNeuralNetworks.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/JuliaGML/HamiltonianNeuralNetworks.jl",
        canonical = "https://juliagml.github.io/HamiltonianNeuralNetworks.jl",
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Hamiltonian Neural Network" => "architectures/hamiltonian_neural_network.md",
        "Generalized Hamiltonian Neural Network" => "architectures/generalized_hamiltonian_neural_network.md",
        "Tutorial" => "tutorials/hamiltonian_neural_network.md",
        "References" => "references.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGML/HamiltonianNeuralNetworks.jl",
    devurl = "latest",
    devbranch = "main"
)
