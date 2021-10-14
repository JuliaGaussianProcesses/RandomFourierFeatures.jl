# The weight_space_approx function needed for ApproximateGPs.pathwise_sample
# Returns both the sampled basis and the distribution over w needed to approximate the GP
function gp_rff_approx(rng, kernel, input_dims, feature_dims)
    ϕ = sample_basis(rng, kernel, input_dims, feature_dims)
    p_w = MvNormal(Diagonal(Fill(1., feature_dims)))
    return ϕ, p_w
end

# Everything necessary to create a RFF approximation to a stationary kernel
# Currently ony supports SqExponentialKernel with variance and lengthscale
struct RFFBasis{Tinner, Touter, Tω, Tτ, Tsample}
    inner_weights::Tinner  # lengthscale
    outer_weights::Touter  # variance (scaled)
    ω::Tω  # Sampled frequencies;               size(ω): (input_dims, num_features)
    τ::Tτ  # Sampled phases;                    size(τ): (num_features,)
    sample_params::Tsample  # Returns a new sample of ω & τ
end

function (ϕ::RFFBasis)(x)
    # ϕ: R^{input_dims} -> R^{num_features}
    x_ = x ./ ϕ.inner_weights
    ωt_x = ϕ.ω'x_
    return ϕ.outer_weights * cos.(ωt_x .+ ϕ.τ)
end

function resample!(ϕ::RFFBasis)
    ω, τ = ϕ.sample_params()
    ϕ.ω .= ω
    ϕ.τ .= τ
end

# Currently need to pass `input_dims` explicitly - will eventually be in KernelFunctions
# https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/16
function sample_basis(rng, kernel, input_dims, num_features=100)
    inner, outer = spectral_weights(kernel)
    outer_scaled = outer * √(2/num_features)
    p_ω = spectral_distribution(kernel, input_dims)

    function sample_params()
        ω = rand(rng, p_ω, num_features)  #
        τ = rand(rng, Uniform(0, 2π), num_features)  #
        return ω, τ
    end

    return RFFBasis(inner, outer_scaled, sample_params()..., sample_params)
end

function spectral_distribution(::SqExponentialKernel, input_dims)
    return MvNormal(Diagonal(Fill(1., input_dims)))
end

spectral_distribution(k::ScaledKernel, args...) = spectral_distribution(k.kernel, args...)
spectral_distribution(k::TransformedKernel, args...) = spectral_distribution(k.kernel, args...)

spectral_distribution(k::Kernel) =  error("Spectral distribution not implemented for kernel:\n$k")

function spectral_weights(::SqExponentialKernel)
    return 1.0, 1.0
end

function spectral_weights(k::ScaledKernel)
    σ² = only(k.σ²)
    inner, outer = spectral_weights(k.kernel)
    return inner, outer * √σ²
end

function spectral_weights(k::TransformedKernel{<:Any, <:ScaleTransform})
    s = only(k.transform.s)
    inner, outer = spectral_weights(k.kernel)
    return inner / s, outer
end

spectral_weights(k::Kernel) = error("Spectral weights not implemented for kernel:\n$k")

# TODO:
# ARDTransform
# ProductKernel
# SumKernel
# MaternKernel
