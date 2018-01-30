α = 0.025

@testset "sub copulas helpers" begin
  Σ = [1 0.5 0.5 0.6; 0.5 1 0.5 0.6; 0.5 0.5 1. 0.6; 0.6 0.6 0.6 1.]
  srand(43)
  x = transpose(rand(MvNormal(Σ),500000))
  y = norm2unifind(x, [1,2])
  @test pvalue(ExactOneSampleKSTest(y[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(y[:,2], Uniform(0,1))) > α
  @test corspearman(y) ≈ [1. 0.; 0. 1.] atol=1.0e-3
  @test makeind(Σ, "clayton" => [1,2]) == [1,2,4]
  x = [0.1 0.2 0.3 0.4; 0.2 0.3 0.4 0.5; 0.2 0.2 0.4 0.4; 0.1 0.3 0.5 0.6]
  @test findsimilar(x, [1,2]) == 4
  srand(43)
  x = randn(100, 4)
  a,b,c = getcors(x,2)
  @test c == [[1, 2]]
  @test b ≈ -0.05690969096909691
  @test a ≈ [0.0976898] atol=1.0e-6
  @test getclust(cor(x), 2)[1] == [2, 3]
  s= [1. 0.2 0.3; 0.2 1. 0.4; 0.3 0.4 1.]
  @test meanΣ(s) == 0.3
end

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

@testset "convert sub-copula" begin
  srand(42)
  Σ = cormatgen(25)
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 500000)'
  y = y.*S'
  d=["clayton" => [1,2,3,4,9]]
  x = gcop2arch(y, d)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(0,S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(0,S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Normal(0,S[4]))) > α
  @test vecnorm(cor(y)-cor(x))/vecnorm(cor(y)) < 0.065
  @test vecnorm(cov(y)-cov(x))/vecnorm(cov(y)) < 0.065
  @test maximum(abs.(cor(y)-cor(x))) < 0.13
  #cg = cumulants(y, 4)
  #c = cumulants(x, 4)
  #@test vecnorm(cg[3]) < 1
  #@test vecnorm(cg[4]) < 1
  #@test vecnorm(c[3]) > 10
  #@test vecnorm(c[4]) > 10
end

@testset "sub copulas based generator" begin
  srand(42)
  Σ = cormatgen(20)
  d=["clayton" => [2,3,4,15,16], "amh" => [1,20], "gumbel" => [9,10], "frank" => [7,8],
  "mo" => [11,12], "frechet" => [5,6,13]]
  srand(42)
  x = copulamix(100000, Σ, d; λ = [6.5, 2.1])
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,9], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,10], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,11], Uniform(0,1))) > α
  λₗ = (2^(-1/ρ2θ(corspearman(x[:,2], x[:,3]), "clayton")))
  λᵣ = (2-2.^(1./ρ2θ(corspearman(x[:,9], x[:,10]), "gumbel")))
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
  d=["gumbel" => [1,2,3,4], "mo" => [5,6,7]]
  srand(44)
  x = copulamix(100000, Σ, d; λ = [10.5, 5.1, 1.1, 20.])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
end

@testset "copula chain based generator" begin
  srand(42)
  Σ = cormatgen(15)
  d=["clayton" => [2,3,4,5,6], "amh" => [1,14], "frank" => [7,8]]
  x = bivariatecopulamix(100000, Σ, d);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
end
