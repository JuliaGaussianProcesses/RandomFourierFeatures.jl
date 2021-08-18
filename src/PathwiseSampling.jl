module PathwiseSampling

using AbstractGPs
using SparseGPs
import AbstractGPs: ApproxPosteriorGP
using Distributions
using LinearAlgebra
using FillArrays
using PDMats

export prior_basis, sample_prior_functions, sample_posterior_functions

include("posterior_sampling.jl")
include("random_fourier_features.jl")

end
