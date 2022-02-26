module RandomFourierFeatures

using KernelFunctions
using BayesianLinearRegressors
using AbstractGPs
using Distributions
using LinearAlgebra
using FillArrays
using Random

export sample_rff_basis, build_rff_weight_space_approx

include("random_fourier_features.jl")

end
