function sample_prior_functions(ϕ, n_features=10, n_samples=1)
    w = rand(Normal(), (n_features, n_samples))
    function f(x)  # output: [n_samples × 1]
        return w'ϕ(x)
    end
    return f, w
end


function sample_posterior_functions(ap::ApproxPosteriorGP{SVGP}, n_features=10, n_samples=1)
    dims = length(ap.data.u[1])

    # 1. sample a prior basis
    ϕ = prior_basis(ap.prior.kernel, n_features, dims)

    # 2. sample a prior function
    f, w = sample_prior_functions(ϕ, n_features, n_samples)

    # 3. pathwise update
    z = ap.data.u
    Pw = w'hcat(ϕ.(z)...)
    q = _construct_q(ap)
    u = rand(q, n_samples)
    v = ap.data.Kuu \ (u - Pw')
    function pathwise_update(x)
        if size(x) == (); x = [x] end
        return v'cov(ap.prior, z, x)
    end

    return x -> f(x) + pathwise_update(x)
end


function _construct_q(gp::ApproxPosteriorGP{SVGP})
    m, A = gp.data.m, gp.data.A
    # PDMats bug: PDMat(Cholesky(Diagonal(X)))
    # https://github.com/JuliaStats/PDMats.jl/issues/137
    return MvNormal(m, Matrix(A))
end
