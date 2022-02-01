α = 0.0225

@testset "helpers" begin
  Σ = [1 0.5 0.5 0.6; 0.5 1 0.5 0.6; 0.5 0.5 1. 0.6; 0.6 0.6 0.6 1.]
  Random.seed!(43)
  x = Array(transpose(rand(MvNormal(Σ),500000)))
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
  Random.seed!(42)
  @test frechet(0.5, [1. 0.2; 0.4 .6]; rng = Random.GLOBAL_RNG) == [1.0  0.2; 0.6 0.6]
  Random.seed!(42)
  x = rand(1000, 5)
  c = getcors_advanced(x)
  if VERSION <= v"1.7"
    @test c[1] == [[1, 4], [2, 3, 5]]
    @test c[2] ≈ [0.04729, 0.0195683] atol=1.0e-4
    @test c[3] ≈ -0.02177 atol=1.0e-4
  else
    @test c[1] == [[2, 3], [4, 5]]
    @test c[2] ≈ [0.009164, 0.0894643]  atol=1.0e-4
    @test c[3] ≈ -0.00568 atol=1.0e-4
  end
  x = frechet(0.6, rand(100000, 4); rng = Random.GLOBAL_RNG)
  Σ = cor(x)
  @test Σ[1,2] ≈ 0.6 atol=1.0e-2
  @test Σ[3,2] ≈ 0.6 atol=1.0e-2
  @test tail(x[:,1], x[:,2], "r") ≈ 0.6 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l") ≈ 0.6 atol=1.0e-1
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
end

@testset "convert sub-copula to archimedean" begin
  Random.seed!(43)
  Σ = cormatgen(25)
  Σ1 =0.8*ones(25,25) + 0.2*Matrix(1.0I, 25, 25)
  Σ = 0.3*Σ + 0.7*Σ1
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 50_000)'
  y = y.*S'

  @testset "Clayton nested, not nested and naive" begin
    d=["clayton" => [1,2,3,4,9]]
    Random.seed!(42)
    x = gcop2arch(y, d)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Normal(0,S[4]))) > α
    @test norm(cor(y)-cor(x))/norm(cor(y)) < 0.045
    @test norm(cov(y)-cov(x))/norm(cov(y)) < 0.045
    @test maximum(abs.(cor(y)-cor(x))) < 0.11

    x1 = gcop2arch(y, d; notnested = true)
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,4], Normal(0,S[4]))) > α
    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.045
    @test norm(cov(y)-cov(x1))/norm(cov(y)) < 0.045
    @test maximum(abs.(cor(y)-cor(x1))) < 0.11

    x2 = gcop2arch(y, d; naive = true)
    @test pvalue(ExactOneSampleKSTest(x2[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x2[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x2[:,4], Normal(0,S[4]))) > α
    @test maximum(abs.(cov(y[:,1:4])-cov(x2[:,1:4]))) < 0.06
    @test_throws AssertionError gcop2arch(y, ["clayton" => [1,1,3,4]])
  end

  @testset "Gumbel not nested and nested" begin
    Random.seed!(42)
    x1 = gcop2arch(y, ["gumbel" => [1,2,3,4,9]]; notnested = true)
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,4], Normal(0,S[4]))) > α
    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.045
    @test norm(cov(y)-cov(x1))/norm(cov(y)) < 0.045
    @test maximum(abs.(cor(y)-cor(x1))) < 0.11

    Random.seed!(42)
    x1 = gcop2arch(y, ["gumbel" => [1,2,3,4,9]])
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,4], Normal(0,S[4]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,9], Normal(0,S[9]))) > α

    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.04
    @test norm(cov(y)-cov(x1))/norm(cov(y)) < 0.04
    @test maximum(abs.(cor(y)-cor(x1))) < 0.1
  end

  @testset "AMH" begin
    Random.seed!(42)
    x1 = gcop2arch(y, ["amh" => [1,2,3]]; notnested = true)
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α

    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.118
    @test maximum(abs.(cor(y)-cor(x1))) < 0.25

    x1 = gcop2arch(y, ["amh" => [1,2,3]])
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α

    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.118
    @test maximum(abs.(cor(y)-cor(x1))) < 0.25
  end

  @testset "Frank" begin
    Random.seed!(42)
    x1 = gcop2arch(y, ["frank" => [1,2,3,4,9]]; notnested = true)
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,4], Normal(0,S[4]))) > α
    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.06
    @test norm(cov(y)-cov(x1))/norm(cov(y)) < 0.06
    @test maximum(abs.(cor(y)-cor(x1))) < 0.14

    x1 = gcop2arch(y, ["frank" => [1,2,3,4,9]])
    @test pvalue(ExactOneSampleKSTest(x1[:,1], Normal(0,S[1]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,3], Normal(0,S[3]))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,4], Normal(0,S[4]))) > α
    @test norm(cor(y)-cor(x1))/norm(cor(y)) < 0.07
    @test norm(cov(y)-cov(x1))/norm(cov(y)) < 0.07
    @test maximum(abs.(cor(y)-cor(x1))) < 0.14
  end
end

@testset "convert sub-copula to t-Student" begin
  Random.seed!(42)
  Σ = cormatgen(25)
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  mu = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 100000)'
  y = y.*S'.+mu'
  x = gcop2tstudent(y, [1,2,3,4], 10)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(mu[3],S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Normal(mu[4],S[4]))) > α
  @test norm(cor(y)-cor(x))/norm(cor(y)) < 0.015
  @test norm(cov(y)-cov(x))/norm(cov(y)) < 0.015
  @test maximum(abs.(cor(y)-cor(x))) < 0.02
  @test_throws AssertionError gcop2tstudent(y, [1,1,3,4], 10)
  x2 = gcop2tstudent(y, [1,2,3,4], 10; naive = true)
  @test pvalue(ExactOneSampleKSTest(x2[:,3], Normal(mu[3],S[3]))) > α
  @test pvalue(ExactOneSampleKSTest(x2[:,4], Normal(mu[4],S[4]))) > α
  @test maximum(abs.(cov(y[:,1:4])-cov(x2[:,1:4]))) < 0.0055
end

@testset "convert sub-copula to Frechet" begin
  Random.seed!(12)
  Σ = cormatgen(25)
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  mu = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 1_000)'
  y = y.*S'.+mu'
  x = gcop2frechet(y, [1,2,3])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Normal(mu[2],S[2]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(mu[3],S[3]))) > α
  @test norm(cor(y)-cor(x))/norm(cor(y)) < 0.21
  @test norm(cov(y)-cov(x))/norm(cov(y)) < 0.2
  @test maximum(abs.(cor(y)-cor(x))) < 0.37
  @test_throws AssertionError gcop2frechet(y, [1,1,3,4])
  Random.seed!(42)
  x2 = gcop2frechet(y, [1,2,3,4]; naive = true)
  @test pvalue(ExactOneSampleKSTest(x2[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x2[:,2], Normal(mu[2],S[2]))) > α
  @test pvalue(ExactOneSampleKSTest(x2[:,3], Normal(mu[3],S[3]))) > α
  @test maximum(abs.(cov(y[:,1:4])-cov(x2[:,1:4]))) < 0.15
end


@testset "convert sub-copula to Marshall-Olkin" begin
  Random.seed!(42)
  Σ = cormatgen(25)
  S = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  mu = rand([0.8, 0.9, 1, 1.1, 1.2], 25)
  y = rand(MvNormal(Σ), 100_000)'
  y = y.*S'.+mu'
  x = gcop2marshallolkin(y, [1,2])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Normal(mu[2],S[2]))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Normal(mu[3],S[3]))) > α
  @test maximum(abs.(cov(y[:,1:2])-cov(x[:,1:2]))) < 0.03
  @test maximum(abs.(cor(y[:,1:2])-cor(x[:,1:2]))) < 0.03
  @test_throws AssertionError gcop2marshallolkin(y, [1,1])
  @test_throws DomainError gcop2marshallolkin(y, [1,2], 1., -1.)
  @test_throws DomainError gcop2marshallolkin(y, [1,2], -1., 1.)
  @test_throws AssertionError gcop2marshallolkin(y, [1,3,4])
  Random.seed!(42)
  x2 = gcop2marshallolkin(y, [1,2]; naive = true)
  @test pvalue(ExactOneSampleKSTest(x2[:,1], Normal(mu[1],S[1]))) > α
  @test pvalue(ExactOneSampleKSTest(x2[:,2], Normal(mu[2],S[2]))) > α
  @test pvalue(ExactOneSampleKSTest(x2[:,3], Normal(mu[3],S[3]))) > α
  @test maximum(abs.(cov(y[:,1:2])-cov(x2[:,1:2]))) < 0.05
end
