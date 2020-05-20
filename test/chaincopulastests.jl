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
    @test simulate_copula(1, Chain_of_Archimedeans([4., 11.], "frank")) ≈ [0.180975  0.492923  0.679345] atol=1.0e-5
    Random.seed!(43)
    @test simulate_copula(1, Chain_of_Archimedeans([4., 11.], ["frank", "clayton"])) ≈ [0.180975  0.492923  0.600322] atol=1.0e-5

    Random.seed!(43)
    u = zeros(1,3)
    simulate_copula!(u, Chain_of_Archimedeans([4., 11.], "frank"))
    @test u ≈ [0.180975  0.492923  0.679345] atol=1.0e-5

    Random.seed!(43)
    u = zeros(1,3)
    simulate_copula!(u, Chain_of_Archimedeans([4., 11.], ["frank", "clayton"]))
    @test u ≈ [0.180975  0.492923  0.600322] atol=1.0e-5
  end

  @testset "larger example" begin
    cops = ["clayton", "clayton", "clayton", "frank", "amh", "amh"]
    #test old dispatching
    Random.seed!(43)
    x = simulate_copula(1000, Chain_of_Archimedeans([-0.9, 3., 2, 4., -0.3, 1.], cops))
    Random.seed!(43)
    x1 = chaincopulagen(1000, [-0.9, 3., 2, 4., -0.3, 1.], cops)
    Random.seed!(43)
    x2 = chaincopulagen(1000, [-0.9, 3., 2, 4., -0.3, 1.], cops; rev = true)
    @test norm(x-x1) ≈ 0.
    @test norm((1 .- x) - x2) ≈ 0.

    Random.seed!(43)
    x = simulate_copula(300000, Chain_of_Archimedeans([-0.9, 3., 2, 4., -0.3, 1.], cops))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,7], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,6], x[:,7], "l") ≈ 0.5 atol=1.0e-1
    @test tail(x[:,5], x[:,6], "r", 0.0001) ≈ 0
    @test tail(x[:,3], x[:,4], "l") ≈ 1/(2^(1/2)) atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r", 0.0001) ≈ 0
    @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-3
    @test corkendall(x)[2,3] ≈ 3/(2+3) atol=1.0e-3

    # further testing
    Random.seed!(43)
    x = simulate_copula(500000, Chain_of_Archimedeans([-0.9, 2.], "clayton"))
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test tail(x[:,2], x[:,3], "l") ≈ 1/(2^(1/2)) atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0

  end
end
@testset "correlations" begin
  Random.seed!(43)
  c = Chain_of_Archimedeans([0.6, -0.2], "clayton", SpearmanCorrelation)
  x = simulate_copula(500000, c)
  @test corspearman(x[:,1], x[:,2]) ≈ 0.6 atol=1.0e-2
  @test corspearman(x[:,2], x[:,3]) ≈ -0.2 atol=1.0e-2

  Random.seed!(43)
  c = Chain_of_Archimedeans([0.6, -0.2], ["clayton", "clayton"], KendallCorrelation)
  x = simulate_copula(500000, c)
  @test corkendall(x[:,1], x[:,2]) ≈ 0.6 atol=1.0e-3
  @test corkendall(x[:,2], x[:,3]) ≈ -0.2 atol=1.0e-3
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
    @test simulate_copula(1, Chain_of_Frechet([0.6, 0.4], [0.3, 0.5])) ≈ [0.888934  0.775377  0.180975] atol=1.0e-5
  end
  @testset "test on large example" begin
    # test old dispatching
    Random.seed!(43)
    x2 = simulate_copula(1000, Chain_of_Frechet([0.9, 0.6, 0.2]))
    Random.seed!(43)
    x1 = chainfrechetcopulagen(1000, [0.9, 0.6, 0.2])
    @test norm(x2 - x1) ≈ 0.

    # one parameter case
    Random.seed!(43)
    x = simulate_copula(500000, Chain_of_Frechet([0.9, 0.6, 0.2]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test corspearman(x) ≈ [1. 0.9 0.6 0.2; 0.9 1. 0.6 0.2; 0.6 0.6 1. 0.2; 0.2 0.2 0.2 1.] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "r") ≈ 0.9 atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l") ≈ 0.9 atol=1.0e-1
    @test tail(x[:,1], x[:,4], "r") ≈ 0.2 atol=1.0e-1

    # two parameters case
    Random.seed!(43)
    x = simulate_copula(500000, Chain_of_Frechet([0.8, 0.5], [0.2, 0.3]));
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test corspearman(x) ≈ [1. 0.6 0.2; 0.6 1. 0.2; 0.2 0.2 1.] atol=1.0e-3
    @test tail(x[:,1], x[:,2], "r") ≈ 0.8 atol=1.0e-1
    @test tail(x[:,2], x[:,3], "r") ≈ 0.5 atol=1.0e-1
  end
end
