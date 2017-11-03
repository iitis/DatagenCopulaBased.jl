α = 0.025
#addprocs(10)
#@everywhere using DatagenCopulaBased

@testset "t-student subcopula" begin
  srand(43)
  x = gausscopulagen(3, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.])
  g2tsubcopula!(x, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.], [1,2])
  @test x ≈ [0.558652  0.719921  0.794493; 0.935573  0.922409  0.345177; 0.217512  0.174138  0.123049] atol=1.0e-5
  srand(43)
  y = gausscopulagen(500000, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.])
  g2tsubcopula!(y, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.], [1,2])
  @test pvalue(ExactOneSampleKSTest(y[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(y[:,2], Uniform(0,1))) > α
end

@testset "nested archimedean copulas halpers" begin
  srand(43)
  u = nestedcopulag("clayton", [2, 2], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
  @test u ≈ [0.193949 0.230553 0.515404 0.557686; 0.712034 0.761276 0.190189 0.208867] atol=1.0e-5
  srand(43)
  @test nestedstep("clayton", [0.2 0.8; 0.1 0.7], [0.2, 0.4], 2., 1.5) ≈ [0.374625 0.836357; 0.0381504 0.500485] atol=1.0e-5
end

@testset "nested archimedean copulas exceptions" begin
  @test_throws(AssertionError, testnestedθϕ([2, 2], [2.1, 2.2], 0.5, "gumbel"))
  @test_throws(AssertionError, testnestedθϕ([2, 2], [2.1, 2.2], 3.5, "gumbel"))
  @test_throws(AssertionError, testnestedθϕ([2, 2], [0.8, 1.1], 0.5, "amh"))
  @test_throws(AssertionError, testnestedθϕ([2, 2], [0.8], 0.5, "amh"))
  @test_throws(AssertionError, nestedarchcopulagen(100000, [2, 2], [2., 2.], 0.5, "fran"))
  @test_throws(AssertionError, nestedarchcopulagen(500000, [2.2, 3.6, 1.1], "gumbel"))
  @test_throws(AssertionError, nestedarchcopulagen(500000, [4.2, 3.6, 0.1], "gumbel"))
end

@testset "nested gumbel copula" begin
  @testset "single nested" begin
    srand(44)
    x = nestedarchcopulagen(500000, [2,2], [4.2, 6.1], 2.1, "gumbel", 1)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
    M = [1. 0.7619 0.52380 0.52380 0.52380; 0.7619 1. 0.52380 0.52380 0.52380; 0.52380 0.52380 1. 0.83606 0.52380]
    @test corkendall(x)[1:3,:] ≈ M atol=1.0e-2
    @test tail(x[:,1], x[:,2], "r", 0.01) ≈ 2-2^(1/4.2) atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r", 0.01) ≈ 2-2^(1/2.1) atol=1.0e-1
    @test tail(x[:,1], x[:,5], "r", 0.01) ≈ 2-2^(1/2.1) atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r", 0.01) ≈ 2-2^(1/6.1) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0
  end
  @testset "double nested" begin
    srand(43)
    x = nestedarchcopulagen(200000, [[2,2], [2,2]], [[4.1, 3.8],[5.1, 6.1]], [1.9, 2.4], 1.2, "gumbel")
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1,:] ≈ [1., 0.7560, 0.47368, 0.47368, 1/6, 1/6, 1/6, 1/6] atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r", 0.01) ≈ 2-2^(1/4.1) atol=1.0e-1
    @test tail(x[:,1], x[:,3], "r", 0.01) ≈ 2-2^(1/1.9) atol=1.0e-1
    @test tail(x[:,1], x[:,5], "r", 0.01) ≈ 2-2^(1/1.2) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0
  end
  @testset "hierarchical" begin
    srand(42)
    x = nestedarchcopulagen(500000, [4.2, 3.6, 1.1], "gumbel")
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1:2,:] ≈ [1. 0.7619 0.72222 0.0909; 0.7619 1. 0.72222 0.0909] atol=1.0e-2
    @test tail(x[:,2], x[:,3], "r", 0.01) ≈ 2-2^(1/3.6) atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r", 0.01) ≈ 2-2^(1/1.1) atol=1.0e-2
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
  end
end

@testset "nested Ali-Mikhail-Haq copula" begin
  srand(43)
  x = nestedarchcopulagen(200000, [3, 2], [0.8, 0.7], 0.5, "amh", 2)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  c = corkendall(x)
  @test c[1:3,1] ≈ [1., 0.23373, 0.23373] atol=1.0e-2
  @test c[3:6,4] ≈ [0.1288, 1., 0.19505, 0.1288] atol=1.0e-2
  @test tail(x[:,4], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,4], x[:,5], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "l", 0.0001) ≈ 0
  @test tail(x[:,6], x[:,7], "l", 0.0001) ≈ 0
  @test tail(x[:,6], x[:,7], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
end

@testset "nested Frank copula" begin
  srand(44)
  x = nestedarchcopulagen(250000, [3, 2],  [8., 10.], 2., "frank", 2)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test corkendall(x)[1:4,1] ≈ [1.0, 0.60262, 0.60262, 0.2139] atol=1.0e-2
  @test corkendall(x)[3:5,4] ≈ [0.2139, 1.0, 0.6658] atol=1.0e-2
  @test corkendall(x)[6:7,6] ≈ [1.0, 0.2139] atol=1.0e-2
  @test tail(x[:,4], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,4], x[:,5], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "l", 0.0001) ≈ 0
  @test tail(x[:,6], x[:,7], "r", 0.0001) ≈ 0
  @test tail(x[:,6], x[:,7], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
end

@testset "nested Clayton copula" begin
  srand(43)
  x = nestedarchcopulagen(500000, [2, 3],  [3., 4.], 1.5, "clayton", 2)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
  cc = corkendall(x)
  @test cc[:,1] ≈ [1.0, 3/5, 1.5/3.5, 1.5/3.5, 1.5/3.5, 1.5/3.5, 1.5/3.5] atol=1.0e-1
  @test cc[:,5] ≈ [1.5/3.5, 1.5/3.5, 2/3, 2/3, 1.0, 1.5/3.5, 1.5/3.5] atol=1.0e-1
  @test cc[6,7] ≈ 1.5/3.5 atol=1.0e-2
  @test tail(x[:,4], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "l", 0.01) ≈ 2^(-1/(1.5)) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l", 0.01) ≈ 2^(-1/3) atol=1.0e-1
  @test tail(x[:,4], x[:,5], "l", 0.01) ≈ 2^(-1/4) atol=1.0e-1
  @test tail(x[:,6], x[:,7], "l", 0.01) ≈ 2^(-1/(1.5)) atol=1.0e-1
end


@testset "copula mixture helpers" begin
  Σ = [1 0.5 0.5 0.6; 0.5 1 0.5 0.6; 0.5 0.5 1. 0.6; 0.6 0.6 0.6 1.]
  srand(43)
  x = transpose(rand(MvNormal(Σ),500000))
  y = norm2unifind(x, Σ, [1,2])
  @test pvalue(ExactOneSampleKSTest(y[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(y[:,2], Uniform(0,1))) > α
  @test cor(y)≈ [1. 0.; 0. 1.] atol=1.0e-3
  @test makeind(Σ, "clayton" => [1,2]) == [1,2,4]
  x = [0.1 0.2 0.3 0.4; 0.2 0.3 0.4 0.5; 0.2 0.2 0.4 0.4; 0.1 0.3 0.5 0.6]
  @test findsimilar(x, [1,2]) == [4]
  srand(43)
  @test getclust(randn(4,100)) == [1,1,1,1]
end

@testset "copula mixture" begin
  srand(44)
  Σ = cormatgen(20, 0.5, false, false)
  d=["clayton" => [2,3,4,15,16], "amh" => [1,20], "gumbel" => [9,10], "frank" => [7,8],
  "mo" => [11,12], "frechet" => [5,6,13]]
  srand(44)
  x = copulamix(100000, Σ, d; λ = [2.5, 3.1, 20.])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,8], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,9], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,10], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,11], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,12], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,15], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,20], Uniform(0,1))) > α
  λₗ = (2^(-1/ρ2θ(Σ[2,3], "clayton")))
  λᵣ = (2-2.^(1./ρ2θ(Σ[9,10], "gumbel")))
  λamh = (Σ[1,20] >= 0.5)? 0.5 : 0.
  @test tail(x[:,2], x[:,3], "l") ≈ λₗ atol=1.0e-1
  @test tail(x[:,5], x[:,6], "l") ≈ Σ[5,6] + 0.1 atol=1.0e-1
  @test tail(x[:,5], x[:,6], "r") ≈ Σ[5,6] + 0.1 atol=1.0e-1
  @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0 atol=1.0e-2
  @test tail(x[:,1], x[:,20], "l") ≈ λamh atol=1.0e-1
  @test tail(x[:,1], x[:,20], "r", 0.0001) ≈ 0 atol=1.0e-2
  @test tail(x[:,9], x[:,10], "r") ≈ λᵣ atol=1.0e-1
  @test tail(x[:,9], x[:,10], "l", 0.0001) ≈ 0 atol=1.0e-1
  @test tail(x[:,7], x[:,8], "r", 0.0001) ≈ 0 atol=1.0e-2
  @test tail(x[:,7], x[:,8], "l", 0.0001) ≈ 0 atol=1.0e-2
  println(vecnorm(Σ))
  println(vecnorm(cor(quantile(Normal(0,1.), x))))
  d=["gumbel" => [1,2,3,4], "mo" => [5,6,7]]
  srand(44)
  x = copulamix(100000, Σ, d; λ = [2.5, .7, 1.1, 20.])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
  println(vecnorm(Σ))
  println(vecnorm(cor(quantile(Normal(0,1.), x))))
end
