α = 0.025

@testset "helpers" begin
  @test rand2cop(0.5, 2., "clayton", 0.5) ≈ 0.5463906428428872
  @test rand2cop(0.5, 2., "frank", 0.5) ≈ 0.5
  @test rand2cop(0.5, -.5, "amh", 0.5) ≈ 0.5061130556252271
end

@testset "exceptions" begin
  @test_throws DomainError testbivθ(-2., "clayton")
  @test_throws DomainError usebivρ(-.9, "amh", SpearmanCorrelation)
  @test_throws DomainError usebivρ(-.25, "amh", KendallCorrelation)
  @test_throws AssertionError Chain_of_Archimedeans([2., 3.], ["frank", "gumbel"])
  @test_throws BoundsError Chain_of_Archimedeans([2., 3., 4.], ["frank", "frank"])
  @test_throws DomainError Chain_of_Archimedeans([2., -3.], ["frank", "clayton"])
  @test_throws AssertionError Chain_of_Archimedeans([2., 3., 4.], "gumbel")
  @test_throws DomainError Chain_of_Archimedeans([2., -3.], "clayton")
  @test_throws DomainError Chain_of_Archimedeans([2., -.3], "clayton", KendallCorrelation)
  @test_throws DomainError Chain_of_Archimedeans([.2, -1.3], ["frank", "clayton"], KendallCorrelation)
  @test_throws AssertionError Chain_of_Archimedeans([0.2, 0.2], "gumbel", KendallCorrelation)
  @test_throws AssertionError Chain_of_Archimedeans([.2, .3], ["gumbel", "clayton"], KendallCorrelation)

  u = zeros(2,3)
  c = Chain_of_Archimedeans([2.], ["clayton"])
  @test_throws AssertionError simulate_copula!(u, c)
end

@testset "chain of Archimedean copulas" begin
  @testset "small example" begin
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, Chain_of_Archimedeans([4., 11.], "frank"); rng = rng) ≈ [0.1810 0.1915 0.2607] atol=1.0e-3
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, Chain_of_Archimedeans([4., 11.], ["frank", "clayton"]); rng = rng) ≈ [0.18102 0.191519 0.20613] atol=1.0e-5

    Random.seed!(43)
    u = zeros(1,3)
    rng = StableRNG(123)
    simulate_copula!(u, Chain_of_Archimedeans([4., 11.], "frank"); rng = rng)
    @test u ≈ [0.18102 0.19151 0.2608] atol=1.0e-3

    Random.seed!(43)
    u = zeros(1,3)
    rng = StableRNG(123)
    simulate_copula!(u, Chain_of_Archimedeans([4., 11.], ["frank", "clayton"]); rng = rng)
    @test u ≈ [0.18102 0.19151 0.20613] atol=1.0e-3
  end

  @testset "larger example" begin
    cops = ["clayton", "clayton", "clayton", "frank", "amh", "amh"]

    Random.seed!(43)
    x = simulate_copula(50_000, Chain_of_Archimedeans([-0.9, 3., 2, 4., -0.3, 1.], cops))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,6], x[:,7], "l") ≈ 0.5 atol=1.0e-1
    @test tail(x[:,5], x[:,6], "r", 0.0001) ≈ 0
    @test tail(x[:,3], x[:,4], "l") ≈ 1/(2^(1/2)) atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r", 0.0001) ≈ 0
    @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-2
    @test corkendall(x)[2,3] ≈ 3/(2+3) atol=1.0e-2

    # further testing
    Random.seed!(43)
    x = simulate_copula(100_000, Chain_of_Archimedeans([-0.9, 2.], "clayton"))
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test tail(x[:,2], x[:,3], "l") ≈ 1/(2^(1/2)) atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0

  end
end
@testset "correlations" begin
  Random.seed!(43)
  c = Chain_of_Archimedeans([0.6, -0.2], "clayton", SpearmanCorrelation)
  x = simulate_copula(50_000, c)
  @test corspearman(x[:,1], x[:,2]) ≈ 0.6 atol=1.0e-2
  @test corspearman(x[:,2], x[:,3]) ≈ -0.2 atol=1.0e-2

  Random.seed!(43)
  c = Chain_of_Archimedeans([0.6, -0.2], ["clayton", "clayton"], KendallCorrelation)
  x = simulate_copula(50_000, c)
  @test corkendall(x[:,1], x[:,2]) ≈ 0.6 atol=1.0e-2
  @test corkendall(x[:,2], x[:,3]) ≈ -0.2 atol=1.0e-2
end


@testset "chain of Frechet copulas" begin
  @testset "exceptions" begin
      @test_throws DomainError Chain_of_Frechet([1.1, 0.8])
      @test_throws DomainError Chain_of_Frechet([-0.1, 0.8])
      @test_throws DomainError Chain_of_Frechet([0.1, 0.1], [-0.1, 0.8])
      @test_throws DomainError Chain_of_Frechet([0.1, 0.3], [0.1, 0.8])
      @test_throws DomainError Chain_of_Frechet([0.1, -0.3], [0.1, 1.2])
      @test_throws AssertionError Chain_of_Frechet([0.1, 0.3], [0.1, 0.2, 0.3])
  end
  @testset "test on small example" begin
    @test fncopulagen([0.2, 0.4], [0.1, 0.1], [0.2 0.4 0.6; 0.3 0.5 0.7]) == [0.6 0.4 0.2; 0.7 0.5 0.3]
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, Chain_of_Frechet([0.6, 0.4], [0.3, 0.5]); rng = rng) ≈ [0.6690 0.3674 0.1810] atol=1.0e-3
  end
  @testset "test on large example" begin

    # one parameter case
    Random.seed!(43)
    x = simulate_copula(100_000, Chain_of_Frechet([0.9, 0.6, 0.2]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test corspearman(x) ≈ [1. 0.9 0.6 0.2; 0.9 1. 0.6 0.2; 0.6 0.6 1. 0.2; 0.2 0.2 0.2 1.] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "r") ≈ 0.9 atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l") ≈ 0.9 atol=1.0e-1
    @test tail(x[:,1], x[:,4], "r") ≈ 0.2 atol=1.0e-1

    # two parameters case
    Random.seed!(43)
    x = simulate_copula(100_000, Chain_of_Frechet([0.8, 0.5], [0.2, 0.3]));
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test corspearman(x) ≈ [1. 0.6 0.2; 0.6 1. 0.2; 0.2 0.2 1.] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "r") ≈ 0.8 atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r") ≈ 0.5 atol=1.0e-1
  end
end

@testset "Big Float implementation" begin

  cops = ["clayton", "clayton", "clayton", "frank", "amh", "amh"]

  Random.seed!(43)
  θs = BigFloat.([-0.9, 3., 2, 4., -0.3, 1.])
  x = simulate_copula(500, Chain_of_Archimedeans(θs, cops))
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
  @test typeof(x) == Array{BigFloat,2}

  va = BigFloat.([0.9, 0.6, 0.2])
  Random.seed!(43)
  x = simulate_copula(1000, Chain_of_Frechet(va))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α

  va = BigFloat.([0.79999, 0.5])
  vb = BigFloat.([0.2, 0.3])
  Random.seed!(43)
  x = simulate_copula(1000, Chain_of_Frechet(va, vb))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α

end
