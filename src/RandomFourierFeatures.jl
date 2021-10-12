module RandomFourierFeatures

using KernelFunctions
using Distributions
using LinearAlgebra
using FillArrays

export sample_basis, create_prior_sample_function

include("posterior_sampling.jl")
include("random_fourier_features.jl")

end
