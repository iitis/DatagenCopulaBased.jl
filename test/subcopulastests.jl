α = 0.025

@testset "helpers" begin
  Σ = [1 0.5 0.5 0.6; 0.5 1 0.5 0.6; 0.5 0.5 1. 0.6; 0.6 0.6 0.6 1.]
  srand(43)
  x = transpose(rand(MvNormal(Σ),500000))
  y = norm2unifind(x, [1,2], "frechet")
  @test pvalue(ExactOneSampleKSTest(y[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(y[:,2], Uniform(0,1))) > α
  @test corspearman(y) ≈ [1. 0.; 0. 1.] atol=1.0e-3
  y = norm2unifind(x, [1,2])
  @test pvalue(ExactOneSampleKSTest(y[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(y[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(y[:,3], Uniform(0,1))) > α
  @test corspearman(y) ≈ [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] atol=1.0e-3
  s= [1. 0.4 0.3; 0.4 1. 0.2; 0.3 0.2 1.]
  @test meanΣ(s) == 0.3
  @test mean_outer(s, [[1,2]]) == 0.25
  p = parameters(s, [[1,2]])
  @test p == ([0.4], 0.25)
  @test are_parameters_good(p...) == true
  @test Σ_theor([0.5], 0.3, [[1,2], [4]]) == [1.0  0.5  0.3; 0.5  1.0  0.3; 0.3  0.3  1.0]
  srand(42)
  @test frechet(0.5, [1. 0.2; 0.4 .6]) == [1.0  0.2; 0.6 0.6]
  srand(42)
  x = rand(1000, 5)
  c = getcors_advanced(x)
  @test c[1] == [[1, 4], [2, 3, 5]]
  @test c[2] ≈ [0.04729, 0.0195683] atol=1.0e-4
  @test c[3] ≈ -0.021774209774209772
  x = frechet(0.6, rand(50000, 4))
  Σ = cor(x)
  @test Σ[1,2] ≈ 0.6 atol=1.0e-2
  @test Σ[3,2] ≈ 0.6 atol=1.0e-2
  @test tail(x[:,1], x[:,2], "r") ≈ 0.6 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l") ≈ 0.6 atol=1.0e-1
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
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

@testset "convert sub-copula to archimedean" begin
  srand(42)
  Σ = cormatgen(25)
  Σ1 =0.8*ones(25,25) + 0.2*eye(25)
  Σ = 0.3*Σ + 0.7*Σ1
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 100000)'
  y = y.*S'
  d=["clayton" => [1,2,3,4,9]]
  x = gcop2arch(y, d)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(0,S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(0,S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Normal(0,S[4]))) > α
  @test vecnorm(cor(y)-cor(x))/vecnorm(cor(y)) < 0.045
  @test vecnorm(cov(y)-cov(x))/vecnorm(cov(y)) < 0.045
  @test maximum(abs.(cor(y)-cor(x))) < 0.11
  x1 = gcop2arch(y, d; notnested = true)
  @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x1[:,4], Normal(0,S[4]))) > α
  @test vecnorm(cor(y)-cor(x1))/vecnorm(cor(y)) < 0.045
  @test vecnorm(cov(y)-cov(x1))/vecnorm(cov(y)) < 0.045
  @test maximum(abs.(cor(y)-cor(x1))) < 0.11
end

@testset "convert sub-copula to t-Student" begin
  srand(42)
  Σ = cormatgen(25)
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  mu = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 100000)'
  y = y.*S'.+mu'
  x = gcop2tstudent(y, [1,2,3,4], 10)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(mu[3],S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Normal(mu[4],S[4]))) > α
  @test vecnorm(cor(y)-cor(x))/vecnorm(cor(y)) < 0.015
  @test vecnorm(cov(y)-cov(x))/vecnorm(cov(y)) < 0.015
  @test maximum(abs.(cor(y)-cor(x))) < 0.02
  @test_throws AssertionError gcop2tstudent(y, [1,1,3,4], 10)
end

@testset "convert sub-copula to Frechet" begin
  srand(42)
  Σ = cormatgen(25)
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  mu = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 100000)'
  y = y.*S'.+mu'
  x = gcop2frechet(y, [1,2,3])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(mu[3],S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Normal(mu[4],S[4]))) > α
  @test vecnorm(cor(y)-cor(x))/vecnorm(cor(y)) < 0.15
  @test vecnorm(cov(y)-cov(x))/vecnorm(cov(y)) < 0.15
  @test maximum(abs.(cor(y)-cor(x))) < 0.25
  @test_throws AssertionError gcop2frechet(y, [1,1,3,4])
end
