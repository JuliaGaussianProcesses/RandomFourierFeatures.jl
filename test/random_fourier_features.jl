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

    function test_kernel_approx(rng, kernel, x, y, input_dims, num_feats)
        ϕ = sample_rff_basis(rng, kernel, input_dims, num_feats)
        km_true = kernelmatrix(kernel, x, y)
        km_approx = kernelmatrix(ϕ, x, y)
        @test km_true ≈ km_approx rtol = 1e-3 atol = 1e-3
    end

    @testset "SqExponentialKernel" begin
        rng = Random.MersenneTwister(54321)
        se_kernel = SqExponentialKernel()
        test_kernel_approx(rng, se_kernel, x_colvecs, y_colvecs, input_dims, num_feats)
    end
    @testset "ScaledKernel" begin
        rng = Random.MersenneTwister(54321)
        scaled_kernel = 2 * SqExponentialKernel()
        test_kernel_approx(rng, scaled_kernel, x_colvecs, y_colvecs, input_dims, num_feats)
    end
    @testset "ScaleTransform" begin
        rng = Random.MersenneTwister(54321)
        st_kernel = SqExponentialKernel() ∘ ScaleTransform(4.0)
        test_kernel_approx(rng, st_kernel, x_colvecs, y_colvecs, input_dims, num_feats)
    end
    @testset "ColVecs/RowVecs" begin
        rng = Random.MersenneTwister(54321)
        k = 3 * (SqExponentialKernel() ∘ ScaleTransform(3.0))
        ϕ = sample_rff_basis(rng, k, input_dims, 10)
        @test ϕ(x_colvecs) == ϕ(x_rowvecs)
        @test kernelmatrix(ϕ, x_colvecs) == kernelmatrix(ϕ, x_rowvecs)
        @test kernelmatrix(ϕ, x_colvecs, y_colvecs) == kernelmatrix(ϕ, x_rowvecs, y_colvecs)
        @test kernelmatrix(ϕ, x_colvecs, y_colvecs) == kernelmatrix(ϕ, x_colvecs, y_rowvecs)
    end
    @testset "rff_weight_space_approx" begin
        rng = Random.MersenneTwister(54321)
        k = 2 * (SqExponentialKernel() ∘ ScaleTransform(5.0))
        f = GP(k)
        rff_wsa = build_rff_weight_space_approx(rng, input_dims, num_feats)
        bfr = rff_wsa(f)
        @test cov(bfr(x_colvecs)) ≈ kernelmatrix(k, x_colvecs) rtol = 1e-3 atol = 1e-3

        f2 = GP(2.0, k)
        @test_throws ErrorException rff_wsa(f2)
    end
end
