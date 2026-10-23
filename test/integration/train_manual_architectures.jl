using HamiltonianNeuralNetworks
using GeometricMachineLearning: Adam, Batch, DataLoader, NeuralNetwork, TrainingOptimizer,
                                params, train!
using Test
import Random

include("../helpers/eltypes.jl")

# Each architecture that the manual trains, with the data shape and the batch its tutorial uses. The
# list holds a `SymmetricMatrix` leaf (`LASympNet`, `LinearSymplecticTransformer`), the strictly
# triangular leaves (`VolumePreservingFeedForward`, `VolumePreservingTransformer`), a Stiefel leaf
# (`SymplecticAutoencoder`), and the three `Chain`s of the volume-preserving-attention tutorial.
# The last place holds the loss; `nothing` trains on the default loss.
function manual_architectures(T)
    hnn = StandardHamiltonianArchitecture(2, 4)
    (("StandardHamiltonianArchitecture", hnn,
        DataLoader(rand(T, 2, 12), rand(T, 2, 12); suppress_info = true), Batch(4),
        HNNLoss(hnn)),)
end

@testset "train! trains every architecture of the manual ($T)" for T in REAL_ELTYPES
    Random.seed!(6)
    for (name, arch, dl, batch, loss) in manual_architectures(T)
        @testset "$name" begin
            nn = NeuralNetwork(arch, T)
            p₀ = deepcopy(params(nn))
            opt = TrainingOptimizer(nn.params; algorithm = Adam())
            history = loss === nothing ?
                      train!(nn, opt, dl, batch, 2; show_progress = false) :
                      train!(nn, opt, dl, batch, 2; loss, show_progress = false)
            @test length(history) == 2
            @test eltype(history) === T
            @test all(isfinite, history)
            @test params(nn) != p₀
        end
    end
end
