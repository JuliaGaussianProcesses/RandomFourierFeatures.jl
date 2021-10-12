# Everything necessary to create a RFF approximation to a prior GP with a stationary kernel
# Currently ony supports SqExponentialKernel with variance and lengthscale
struct PriorBasis
    inner_weights  # lengthscale
    outer_weights  # variance (scaled)
    ω  # Sampled frequencies
    τ  # Sampled phases
end

(ϕ::PriorBasis)(x) = ϕ([x])
function (ϕ::PriorBasis)(x::AbstractVector)
    # size(x): (num_data, input_dims)
    # size(ϕ.ω): (1, num_features)
    x_rescaled = x / ϕ.inner_weights
    ωt_x = x_rescaled * ϕ.ω  # size(ωt_x): (num_data, num_features)
    return ϕ.outer_weights * cos.(ωt_x .+ ϕ.τ')
end

function prior_basis(rng, kernel, num_features=10, input_dims=1)
    p_ω = spectral_distribution(kernel, input_dims)
    ω = rand(rng, p_ω, num_features)
    τ = rand(rng, Uniform(0, 2π), num_features)
    inner, outer = spectral_weights(kernel)
    outer_scaled = outer * √(2/num_features)

    return PriorBasis(inner, outer_scaled, ω, τ)
end


function spectral_distribution(::SqExponentialKernel, input_dims=1)
    return MvNormal(Diagonal(Fill(1., input_dims)))
end

spectral_distribution(k::ScaledKernel, args...) = spectral_distribution(k.kernel, args...)
spectral_distribution(k::TransformedKernel, args...) = spectral_distribution(k.kernel, args...)

spectral_distribution(k::Kernel) =  error("Spectral distribution not implemented for kernel:\n$k")

function spectral_weights(::SqExponentialKernel)
    return 1.0, 1.0
end

function spectral_weights(k::ScaledKernel)
    σ² = k.σ²[1]
    inner, outer = spectral_weights(k.kernel)
    return inner, outer * √σ²
end

function spectral_weights(k::TransformedKernel{<:Any, <:ScaleTransform})
    s = k.transform.s[1]
    inner, outer = spectral_weights(k.kernel)
    return inner / s, outer
end

spectral_weights(k::Kernel) = error("Spectral weights not implemented for kernel:\n$k")

# TODO:
# ARDTransform
# ProductKernel
# SumKernel
# MaternKernel
