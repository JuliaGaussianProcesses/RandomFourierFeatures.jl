# Everything necessary to create a RFF approximation to a prior GP with a stationary kernel
# Currently ony supports SqExponentialKernel with variance and lengthscale
struct RFFBasis{Tinner, Touter, Tω, Tτ, Tsample}
    inner_weights::Tinner  # lengthscale
    outer_weights::Touter  # variance (scaled)
    ω::Tω  # Sampled frequencies
    τ::Tτ  # Sampled phases
    sample_params::Tsample  # Function to resample ω & τ
end

function (ϕ::RFFBasis)(x::AbstractVector)
    # length(x[1]) = input_dims
    # size(ϕ.ω): (input_dims, num_features)
    x_rescaled = x ./ ϕ.inner_weights

    ωt_x = map(s -> ϕ.ω * s, x_rescaled)  # length(ωt_x[1]) = num_features
    cos_term = map(s -> cos.(s .+ ϕ.τ), ωt_x)
    return ϕ.outer_weights * cos_term
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
        ω = rand(rng, p_ω, num_features)'  # size(ω): (num_features, input_dims)
        τ = rand(rng, Uniform(0, 2π), num_features)  # size(τ): (num_features,)
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
