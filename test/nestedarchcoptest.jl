α = 0.025
#addprocs(10)
#@everywhere using DatagenCopulaBased


@testset "nested archimedean copulas helpers" begin
  srand(43)
  u = nestedcopulag("clayton", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
  @test u ≈ [0.153282 0.182421 0.636606 0.679396; 0.381051 0.430175 0.254842 0.279192] atol=1.0e-5
  srand(43)
  n = nestedstep("clayton", [0.2 0.8; 0.1 0.7], [0.2, 0.4], 2., 1.5)
  @test n ≈ [0.0504023 0.545041; 0.0736747 0.58235] atol=1.0e-5
end

@testset "nested archimedean copulas exceptions" begin
  nestedarchcopulagen(100000, [2, 2], [2., 2.], 0.5, "frank")
  @test_throws DomainError testnestedθϕ([2, 2], [2.1, 2.2], 0.5, "gumbel")
  @test_throws DomainError testnestedθϕ([2, 2], [2.1, 2.2], 3.5, "gumbel")
  @test_throws DomainError testnestedθϕ([2, 2], [0.8, 1.1], 0.5, "amh")
  @test_throws AssertionError testnestedθϕ([2, 2], [0.8], 0.5, "amh")
  @test_throws AssertionError nestedarchcopulagen(100000, [2, 2], [2., 2.], 0.5, "fran")
  @test_throws DomainError nestedarchcopulagen(500000, [2.2, 3.6, 1.1], "gumbel")
  @test_throws DomainError nestedarchcopulagen(500000, [4.2, 3.6, 0.1], "gumbel")
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
  srand(42)
  x = nestedarchcopulagen(100000, [2, 3],  [3., 4.], 1.5, "clayton", 2)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
  cc = corkendall(x)
  @test cc[:,1] ≈ [1.0, 3/5, 1.5/3.5, 1.5/3.5, 1.5/3.5, 1.5/3.5, 1.5/3.5] atol=1.0e-2
  @test cc[:,5] ≈ [1.5/3.5, 1.5/3.5, 2/3, 2/3, 1.0, 1.5/3.5, 1.5/3.5] atol=1.0e-2
  @test cc[6,7] ≈ 1.5/3.5 atol=1.0e-2
  @test tail(x[:,4], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,5], "l", 0.01) ≈ 2^(-1/(1.5)) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l", 0.01) ≈ 2^(-1/3) atol=1.0e-1
  @test tail(x[:,4], x[:,5], "l", 0.01) ≈ 2^(-1/4) atol=1.0e-1
  @test tail(x[:,6], x[:,7], "l", 0.01) ≈ 2^(-1/(1.5)) atol=1.0e-1
end
