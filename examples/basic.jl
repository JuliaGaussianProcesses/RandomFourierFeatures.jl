using PathwiseSampling
using AbstractGPs
using SparseGPs
using Distributions
using LinearAlgebra

k = 3 * (SqExponentialKernel() ∘ ScaleTransform(3.0))
gp = GP(k)
x = rand(9)
fx = gp(x, 0.01)
y = rand(fx)
C = rand(length(x), length(x))
ap = approx_posterior(SVGP(), fx, MvNormal(C'C))

n_feats = Int(1e6)
n_samples = 100

ϕ = prior_basis(k, n_feats)
k_approx(ϕ, x, y) = ϕ(x)'ϕ(y)
k_matrix_approx(ϕ, X, Y) = hcat(ϕ.(RowVecs(X))...)'hcat(ϕ.(RowVecs(Y))...)
# k_matrix_approx(ϕ, X, Y) = reshape(hcat(k_approx.(ϕ, X, Y')...), (length(X), length(Y)))
k_matrix_approx(ϕ, X) = k_matrix_approx(ϕ, X, X)

km_true = kernelmatrix(k, x)
km_approx = k_matrix_approx(ϕ, x) # TODO: too slow?

using Test
@test km_true ≈ km_approx rtol=1e-3 atol=1e-3

n_feats = 10
ϕ = prior_basis(k, n_feats)

f, w = sample_prior_functions(ϕ, n_feats, n_samples)

f_post = sample_posterior_functions(ap, n_feats, n_samples)

using Plots
x_plot = sort(rand(100))
plot(x_plot, hcat(f.(x_plot)...)', color=:blue, linealpha=0.1, label="")
plot!(x_plot, gp, label="True prior")

plot(x_plot, hcat(f_post.(x_plot)...)', label="", color=:red, linealpha=0.2)
plot!(x_plot, ap, color=:green, label="True posterior")

# # 2 dimensional input
x2 = RowVecs(rand(9, 2))
ψ = prior_basis(k, 10, 2)

p = hcat(ψ.(x2)...)

p*p'
