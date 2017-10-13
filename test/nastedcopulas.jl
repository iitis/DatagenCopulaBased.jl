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

@testset "copula mixture" begin
  srand(44)
  x ,s = copulamix(100000, 20, false, [2,3,4,5,6], [1,20], [9,10], [7,8], [11,12]);
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
  λₗ = (2^(-1/ρ2θ(s[2,3], "clayton")))
  λᵣ = (2-2.^(1./ρ2θ(s[9,10], "gumbel")))
  λamh = (s[1,20] >= 0.5)? 0.5 : 0.
  @test lefttail(x[:,2], x[:,3]) ≈ λₗ atol=1.0e-1
  @test lefttail(x[:,3], x[:,4]) ≈ λₗ atol=1.0e-1
  @test lefttail(x[:,4], x[:,5]) ≈ λₗ atol=1.0e-1
  @test lefttail(x[:,2], x[:,4]) ≈ λₗ atol=1.0e-1
  @test lefttail(x[:,1], x[:,20]) ≈ λamh atol=1.0e-1
  @test righttail(x[:,1], x[:,20]) ≈ 0 atol=1.0e-1
  @test lefttail(x[:,9], x[:,10]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,9], x[:,10]) ≈ λᵣ atol=1.0e-1
  @test righttail(x[:,7], x[:,8]) ≈ 0 atol=1.0e-1
  @test lefttail(x[:,7], x[:,8]) ≈ 0 atol=1.0e-1
end
