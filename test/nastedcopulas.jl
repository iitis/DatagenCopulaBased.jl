α = 0.025

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

@testset "nasted gumbel copula" begin
  @testset "hierarchical" begin
    srand(43)
    x = nastedgumbelcopula(500000, [4.2, 3.6, 1.1])
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1:2,:] ≈ [1. 0.7619 0.72222 0.0909; 0.7619 1. 0.72222 0.0909] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "r") ≈ 0.8206 atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r") ≈ 0.7877 atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r") ≈ 0.1221 atol=1.0e-2
  end
  @testset "single nasted" begin
    srand(43)
    x = nastedgumbelcopula(600000, [2,2], [4.2, 6.1], 2.1)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1:3,:] ≈ [1. 0.7619 0.52380 0.52380; 0.7619 1. 0.52380 0.52380; 0.52380 0.52380 1. 0.83606] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "r") ≈ 0.82056 atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r") ≈ 0.6089 atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r") ≈ 0.87966 atol=1.0e-1
  end
  @testset "double nasted" begin
    srand(43)
    x = nastedgumbelcopula(200000, [[2,2], [2,2]], [[4.1, 3.8],[5.1, 6.1]], [1.9, 2.4], 1.2)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1,:] ≈ [1., 0.7560, 0.47368, 0.47368, 0.16666, 0.16666, 0.16666, 0.16666] atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r") ≈ 0.8158 atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r") ≈ 0.5597 atol=1.0e-1
    @test tail(x[:,1], x[:,3], "r") ≈ 0.5597 atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r") ≈ 0.7999 atol=1.0e-1
    @test tail(x[:,1], x[:,5], "r") ≈ 0.2182 atol=1.0e-1
  end
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
end


@testset "copula mixture" begin
  srand(44)
  Σ = cormatgen(20, 0.5, false, false)
  d=["clayton" => [2,3,4,5,6], "amh" => [1,20], "gumbel" => [9,10], "frank" => [7,8], "Marshal-Olkin" => [11,12]]
  srand(44)
  x = copulamix(100000, Σ, d)
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
  @test pvalue(ExactOneSampleKSTest(x[:,20], Uniform(0,1))) > α
  λₗ = (2^(-1/ρ2θ(Σ[2,3], "clayton")))
  λᵣ = (2-2.^(1./ρ2θ(Σ[9,10], "gumbel")))
  λamh = (Σ[1,20] >= 0.5)? 0.5 : 0.
  @test tail(x[:,2], x[:,3], "l") ≈ λₗ atol=1.0e-1
  @test tail(x[:,3], x[:,4], "l") ≈ λₗ atol=1.0e-1
  @test tail(x[:,4], x[:,5], "l") ≈ λₗ atol=1.0e-1
  @test tail(x[:,2], x[:,4], "l") ≈ λₗ atol=1.0e-1
  @test tail(x[:,1], x[:,20], "l") ≈ λamh atol=1.0e-1
  @test tail(x[:,1], x[:,20], "r") ≈ 0 atol=1.0e-1
  @test tail(x[:,9], x[:,10], "l") ≈ 0 atol=1.0e-1
  @test tail(x[:,9], x[:,10], "r") ≈ λᵣ atol=1.0e-1
  @test tail(x[:,7], x[:,8], "r") ≈ 0 atol=1.0e-1
  @test tail(x[:,7], x[:,8], "l") ≈ 0 atol=1.0e-1
  d=["gumbel" => [1,2,3,4], "Marshal-Olkin" => [5,6,7]]
  srand(44)
  x = copulamix(100000, Σ, d, [2., 1.8, 1.3, 0.6])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
end
