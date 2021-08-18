# Everything necessary to create a RFF approximation to a prior GP with a stationary kernel
# Currently ony supports SqExponentialKernel with variance and lengthscale
struct PriorBasis
    inner  # lengthscale
    outer  # variance (scaled)
    ω  # Sampled frequencies
    τ  # Sampled phases
end

(ϕ::PriorBasis)(x::Real) = ϕ([x])
function (ϕ::PriorBasis)(x::AbstractVector)
    x_rescaled = x / ϕ.inner
    ωt_x = ϕ.ω' * x_rescaled
    return ϕ.outer * cos.(ωt_x + ϕ.τ)
end

function prior_basis(kernel, n_features=10, dims=1)
    p_ω = spectral_distribution(kernel, dims)
    ω = rand(p_ω, n_features)
    τ = rand(Uniform(0, 2π), n_features)
    inner, outer = spectral_weights(kernel)
    outer_wts = outer * √(2/n_features)
    return PriorBasis(inner, outer_wts, ω, τ)
    # function ϕ(x)
    #     x_rescaled = x / inner
    #     ωt_x = ω' * x_rescaled
    #     return outer_wts * cos.(ωt_x + τ)
    # end
    # return ϕ
end


function spectral_distribution(::SqExponentialKernel, dims=1)
    return MvNormal(Diagonal(Fill(1., dims)))
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
