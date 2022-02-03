@testset "random_fourier_features" begin
    rng = Random.MersenneTwister(54321)

    input_dims = 2
    x = rand(rng, 3, input_dims)
    x_rowvecs = RowVecs(x)
    x_colvecs = ColVecs(x')

    y = rand(rng, 4, input_dims)
    y_rowvecs = RowVecs(y)
    y_colvecs = ColVecs(y')

    num_feats = Int(1e7)

    function test_kernel_approx(kernel, x, y, input_dims, num_feats)
        ϕ = sample_rff_basis(rng, kernel, input_dims, num_feats)
        km_true = kernelmatrix(kernel, x, y)
        km_approx = kernelmatrix(ϕ, x, y)
        @test km_true ≈ km_approx rtol = 1e-3 atol = 1e-3
    end

    @testset "SqExponentialKernel" begin
        se_kernel = SqExponentialKernel()
        test_kernel_approx(se_kernel, x_colvecs, y_colvecs, input_dims, num_feats)
    end
    @testset "ScaledKernel" begin
        scaled_kernel = 2 * SqExponentialKernel()
        test_kernel_approx(scaled_kernel, x_colvecs, y_colvecs, input_dims, num_feats)
    end
    @testset "ScaleTransform" begin
        st_kernel = SqExponentialKernel() ∘ ScaleTransform(4.0)
        test_kernel_approx(st_kernel, x_colvecs, y_colvecs, input_dims, num_feats)
    end
    @testset "ColVecs/RowVecs" begin
        k = 3 * (SqExponentialKernel() ∘ ScaleTransform(3.0))
        ϕ = sample_rff_basis(rng, k, input_dims, 10)
        @test ϕ(x_colvecs) == ϕ(x_rowvecs)
        @test kernelmatrix(ϕ, x_colvecs) == kernelmatrix(ϕ, x_rowvecs)
        @test kernelmatrix(ϕ, x_colvecs, y_colvecs) == kernelmatrix(ϕ, x_rowvecs, y_colvecs)
        @test kernelmatrix(ϕ, x_colvecs, y_colvecs) == kernelmatrix(ϕ, x_colvecs, y_rowvecs)
    end
end
