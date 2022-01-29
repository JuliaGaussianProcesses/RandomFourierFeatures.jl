using RandomFourierFeatures

input_dims = 1

k = 3 * (SqExponentialKernel() ∘ ScaleTransform(3.0))
x = ColVecs(rand(9, input_dims)')

n_feats = Int(1e6)
n_samples = Int(1e6)

ϕ = sample_rff_basis(k, input_dims, n_feats)
k_matrix_approx(ϕ, x, y) = ϕ(x)'ϕ(y)
function k_matrix_approx(ϕ, x)
    ϕx = ϕ(x)
    return ϕx'ϕx
end

km_true = kernelmatrix(k, x)
km_approx = k_matrix_approx(ϕ, x) # TODO: too slow?

using Test
@test km_true ≈ km_approx rtol = 1e-3 atol = 1e-3
