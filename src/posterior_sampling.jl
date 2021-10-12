# Creates the sampling function needed for ApproximateGPs.pathwise_sample
function create_prior_sample_function(num_basis_features=100)
    function prior_sample_function(rng, prior, input_dims, num_samples)
        ϕ = sample_basis(rng, prior.kernel, num_basis_features, input_dims)
        w = rand(rng, Normal(), (num_basis_features, num_samples))
        return ϕ, w
    end
    return prior_sample_function
end

## Now in ApproximateGPs
# struct PosteriorSample
#     prior  # prior(x) returns an approx prior sample
#     update  # update(x) computes the pathwise update
# end

# (f::PosteriorSample)(x) = f([x])
# (f::PosteriorSample)(x::AbstractVector) = f.prior(x) + f.update(x)

# function sample_posterior_functions(f::ApproxPosteriorGP{SVGP}, n_features=10, n_samples=1)
#     dims = length(f.data.u[1])

#     # 1. sample a prior basis
#     ϕ = prior_basis(f.prior.kernel, n_features, dims)

#     # 2. sample a prior function
#     prior, w = sample_prior_functions(ϕ, n_features, n_samples)

#     # 3. pathwise update
#     z = f.data.u
#     Pw = w'hcat(ϕ.(z)...)
#     q = f.approx.q
#     u = rand(q, n_samples)
#     v = f.data.Kuu \ (u - Pw')
#     update(x) = v'cov(f.prior, z, x)

#     return PosteriorSample(prior, update)
# end
