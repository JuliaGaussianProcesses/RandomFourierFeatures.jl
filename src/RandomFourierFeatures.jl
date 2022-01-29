module RandomFourierFeatures

using KernelFunctions
using Distributions
using LinearAlgebra
using FillArrays
using Random

export sample_rff_basis, gp_rff_approx

include("random_fourier_features.jl")

end
