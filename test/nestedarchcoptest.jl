α = 0.025

@testset "nested archimedean copulas helpers 4 higher correlations" begin
  Random.seed!(43)
  rng = StableRNG(123)
  u = nestedcopulag("clayton", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6]; rng = rng)
  @test u ≈ [0.2496 0.2946 0.71634 0.75654; 0.32992 0.37465 0.31906 0.34895] atol=1.0e-3

  Random.seed!(42)
  rng = StableRNG(123)
  x = nestedcopulag("gumbel", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6]; rng = rng)
  @test x ≈ [0.0940 0.13852 0.17391 0.20250; 0.31450 0.3676 0.45506 0.4880] atol=1.0e-3

end

@testset "nested Clayton copula" begin
  @testset "exceptions" begin
    a = ClaytonCopula(2, 1.)
    b = ClaytonCopula(2, 2.)
    c = ClaytonCopula(2, 10.)
    d = ClaytonCopula(2, 20.)

    @test_throws DomainError NestedClaytonCopula([a,b], 0, 1.5)
    @test_throws DomainError NestedClaytonCopula([a,b], -1, 0.5)

    u = zeros(3,5)
    cp = NestedClaytonCopula([a,b], 0, 0.5)
    @test_throws AssertionError simulate_copula!(u, cp)
    #@test_warn "θ << ϕ, marginals may not be uniform" NestedClaytonCopula([c,d], 0, 0.05)
  end
  @testset "small example" begin
      c1 = ClaytonCopula(2, 2.)
      c2 = ClaytonCopula(2, 3.)
      cp = NestedClaytonCopula([c1, c2], 1, 1.1)

      Random.seed!(43)
      rng = StableRNG(123)
      @test simulate_copula(1, cp; rng = rng) ≈ [0.274511 0.349494 0.8443515 0.5447064 0.44514597] atol=1.0e-5


      u = zeros(1,5)
      Random.seed!(43)
      rng = StableRNG(123)
      simulate_copula!(u, cp; rng = rng)
      @test u ≈ [0.274511 0.349494 0.8443515 0.5447064 0.44514597] atol=1.0e-5

  end
  @testset "large example on data" begin
      c1 = ClaytonCopula(2, 3.)
      c2 = ClaytonCopula(3, 4.)
      cp = NestedClaytonCopula([c1, c2], 2, 1.5)


      Random.seed!(42)
      x = simulate_copula(20_000, cp)
      @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
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

      #test on correlations
      c1 = ClaytonCopula(2, .7, KendallCorrelation)
      cp = NestedClaytonCopula([c1], 1, 0.3, KendallCorrelation)
      x = simulate_copula(20_000, cp)
      @test corkendall(x)[:,1] ≈ [1, 0.7, 0.3] atol=1.0e-2
    end
end

@testset "nested Ali-Mikhail-Haq copula" begin
  @testset "exceptions" begin
    a = AmhCopula(2, .8)
    b = AmhCopula(2, .3)
    @test_throws DomainError NestedAmhCopula([a,b], 0, 0.5)
    @test_throws DomainError NestedAmhCopula([a,b], -1, 0.1)

    u = zeros(5, 3)
    cp = NestedAmhCopula([a,b], 0, 0.1)
    @test_throws AssertionError simulate_copula!(u, cp)
  end
  @testset "small example" begin
      c1 = AmhCopula(2, .8)
      c2 = AmhCopula(2, .9)
      cp = NestedAmhCopula([c1, c2], 1, 0.5)
      Random.seed!(43)
      rng = StableRNG(123)
      @test simulate_copula(1, cp; rng = rng) ≈ [0.2066 0.3355 0.7459 0.2496 0.2804] atol=1.0e-3

      u = zeros(1,5)
      Random.seed!(43)
      rng = StableRNG(123)
      simulate_copula!(u, cp; rng = rng)
      @test u ≈[0.2066 0.3355 0.7459 0.2496 0.2804] atol=1.0e-3

  end
  @testset "large example" begin
      c1 = AmhCopula(3, .8)
      c2 = AmhCopula(2, .7)
      cp = NestedAmhCopula([c1, c2], 2, 0.5)


      Random.seed!(44)
      x = simulate_copula(5_000, cp)
      @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
      @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
      c = corkendall(x)
      @test c[1:3,1] ≈ [1., 0.2337, 0.2337] atol=1.0e-2
      @test c[3:6,4] ≈ [0.130, 1., 0.195, 0.129] atol=1.0e-1
      @test tail(x[:,4], x[:,5], "r", 0.0001) ≈ 0
      @test tail(x[:,1], x[:,5], "r", 0.0001) ≈ 0
      @test tail(x[:,4], x[:,5], "l", 0.0001) ≈ 0
      @test tail(x[:,1], x[:,5], "l", 0.0001) ≈ 0
      @test tail(x[:,6], x[:,7], "l", 0.0001) ≈ 0
      @test tail(x[:,6], x[:,7], "r", 0.0001) ≈ 0
      @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
      @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

      #test on correlations
      c1 = AmhCopula(2, .2, KendallCorrelation)
      cp = NestedAmhCopula([c1], 1, 0.1, KendallCorrelation)
      x = simulate_copula(10_000, cp)
      @test corkendall(x)[:,1] ≈ [1, 0.2, 0.1] atol=1.0e-2
  end
end

@testset "nested Frank copula" begin
  @testset "exceptions" begin
    a = FrankCopula(2, 2.)
    b = FrankCopula(2, -1.)
    c = FrankCopula(2, 1.)
    @test_throws DomainError NestedFrankCopula([a,b], 0, 0.5)
    @test_throws DomainError NestedFrankCopula([a,c], -1, 0.1)

    u = zeros(5, 7)
    cp = NestedFrankCopula([a,c], 0, 0.1)
    @test_throws AssertionError simulate_copula!(u, cp)
  end
  @testset "small data set" begin
    a = FrankCopula(2, 2.)
    b = FrankCopula(2, 3.)
    cp = NestedFrankCopula([a,b], 1, 1.1)

    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, cp; rng = rng) ≈ [0.0851 0.1911 0.5927 0.13458 0.31417] atol=1.0e-3

    Random.seed!(43)
    rng = StableRNG(123)
    u = zeros(1,5)
    simulate_copula!(u, cp; rng = rng)
    @test u ≈ [0.0851 0.1911 0.5927 0.13458 0.31417] atol=1.0e-3
  end
  @testset "large data set" begin

    a = FrankCopula(3, 8.)
    b = FrankCopula(2, 10.)
    cp = NestedFrankCopula([a,b], 2, 2.)


    Random.seed!(43)
    x = simulate_copula(5_000, cp)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α

    @test corkendall(x)[1:4,1] ≈ [1.0, 0.60262, 0.60262, 0.2139] atol=1.0e-1
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

    # correlation tests
    c1 = FrankCopula(2, .6, KendallCorrelation)
    cp = NestedFrankCopula([c1], 1, 0.2, KendallCorrelation)
    x = simulate_copula(10_000, cp)
    @test corkendall(x)[:,1] ≈ [1, 0.6, 0.2] atol=1.0e-1
  end
end

@testset "single nested Gumbel" begin
  @testset "exceptios" begin
    a = GumbelCopula(2, 2.)
    b = GumbelCopula(2, 1.5)
    @test_throws DomainError NestedGumbelCopula([a,b], 0, 1.7)
    @test_throws DomainError NestedGumbelCopula([a,b], -1, 1.1)

    u = zeros(5, 7)
    cp = NestedGumbelCopula([a,b], 1, 1.1)
    @test_throws AssertionError simulate_copula!(u, cp)
  end
  @testset "test on small data" begin
    a = GumbelCopula(2, 2.)
    b = GumbelCopula(2, 3.)
    cp = NestedGumbelCopula([a,b], 1, 1.1)
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, cp; rng = rng) ≈ [0.67589 0.740974 0.243834 0.06055 0.34778] atol=1.0e-5


    u = zeros(1,5)
    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u, cp; rng = rng)
    @test u ≈ [0.67589 0.740974 0.243834 0.06055 0.34778] atol=1.0e-5


    Random.seed!(43)
    rng = StableRNG(123)
    n = nestedcopulag("gumbel", [[1,2], [3,4]], [2., 3.], 1.1,  [0.1 0.2 0.3 0.4 0.5; 0.1 0.2 0.3 0.4 0.5]; rng = rng)
    @test n ≈ [0.09401 0.13852 0.17391 0.20250; 0.25067 0.3145 0.42217 0.45506] atol=1.0e-3

  end
  @testset "test on larger data" begin
    a = GumbelCopula(2, 4.2)
    b = GumbelCopula(2, 6.1)
    cp = NestedGumbelCopula([a,b], 1, 2.1)

    Random.seed!(44)
    x = simulate_copula(1_00_000, cp)
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
    @test tail(x[:,1], x[:,2], "l", 0.000001) ≈ 0
    @test tail(x[:,3], x[:,4], "l", 0.000001) ≈ 0
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0
    @test tail(x[:,2], x[:,3], "l", 0.00001) ≈ 0
    @test tail(x[:,1], x[:,5], "l", 0.00001) ≈ 0

    # correlation tests
    c1 = GumbelCopula(2, .8, KendallCorrelation)
    cp = NestedGumbelCopula([c1], 1, 0.2, KendallCorrelation)
    x = simulate_copula(10_000, cp)
    @test corkendall(x)[:,1] ≈ [1, 0.8, 0.2] atol=1.0e-2
  end
end

@testset "double nested Gumbel copula" begin
  @testset "exceptions" begin
    a = GumbelCopula(2, 4.2)
    b = GumbelCopula(2, 6.1)
    cp = NestedGumbelCopula([a,b], 1, 2.1)

    a1 = GumbelCopula(2, 4.2)
    b1 = GumbelCopula(2, 5.1)
    cp1 = NestedGumbelCopula([a1,b1], 0, 3.1)

    @test_throws DomainError DoubleNestedGumbelCopula([cp, cp1], 2.2)
    @test_throws DomainError DoubleNestedGumbelCopula([cp, cp1], 0.9, KendallCorrelation)
    copula = DoubleNestedGumbelCopula([cp, cp1], 1.2)
    u = rand(3,20)
    @test_throws AssertionError simulate_copula!(u, copula)
  end
  @testset "small data" begin
    Random.seed!(43)
    a = GumbelCopula(2, 5.)
    b = GumbelCopula(2, 6.)
    p1 = NestedGumbelCopula([a,b], 1, 2.)

    c = GumbelCopula(2, 5.5)
    p2 = NestedGumbelCopula([c], 2, 2.1)

    copula = DoubleNestedGumbelCopula([p1, p2], 1.5)
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, copula; rng = rng) ≈ [0.4058 0.43187 0.13949 0.08405 0.20398 0.41659 0.46121 0.68961 0.44060]  atol=1.0e-3


    u = zeros(1,9)
    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u, copula; rng = rng)
    #println(u)
    @test u ≈ [0.4058 0.43187 0.13949 0.08405 0.20398 0.41659 0.46121 0.68961 0.44060] atol=1.0e-3


  end
  @testset "large data" begin
    a = GumbelCopula(2, 4.1)
    b = GumbelCopula(2, 3.8)
    cp = NestedGumbelCopula([a,b], 0, 1.9)

    a1 = GumbelCopula(2, 5.1)
    b1 = GumbelCopula(2, 6.1)
    cp1 = NestedGumbelCopula([a1, b1], 0, 2.4)
    cgp = DoubleNestedGumbelCopula([cp, cp1], 1.2)


    Random.seed!(43)
    x = simulate_copula(10_000, cgp)
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

    # correlation tests
    a = GumbelCopula(2, 0.8, KendallCorrelation)
    cp = NestedGumbelCopula([a], 1, 0.5, KendallCorrelation)

    a1 = GumbelCopula(2, 0.7, KendallCorrelation)
    cp1 = NestedGumbelCopula([a1], 1, 0.4, KendallCorrelation)
    cgp = DoubleNestedGumbelCopula([cp, cp1], 0.2, KendallCorrelation)

    x = simulate_copula(10_000, cgp)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,6], Uniform(0,1))) > α
    c1 = [1, 0.8, 0.5, 0.2, 0.2, 0.2]
    c2 = [0.8, 1, 0.5, 0.2, 0.2, 0.2]
    c3 = [0.5, 0.5, 1, 0.2, 0.2, 0.2]
    c4 = [0.2, 0.2, 0.2, 1., 0.7, 0.4]
    c5 = [0.2, 0.2, 0.2, 0.7, 1., 0.4]
    c6 = [0.2, 0.2, 0.2, .4, 0.4, 1.]
    C = corkendall(x)
    @test C[:,1] ≈ c1 atol=1.0e-1
    @test C[:,2] ≈ c2 atol=1.0e-1
    @test C[:,3] ≈ c3 atol=1.0e-1
    @test C[:,4] ≈ c4 atol=1.0e-1
    @test C[:,5] ≈ c5 atol=1.0e-1
    @test C[:,6] ≈ c6 atol=1.0e-1
  end
end

@testset "Hierarchical Gumbel copula" begin
  @testset "exceptions" begin
    @test_throws DomainError HierarchicalGumbelCopula([5., 6., 7.])
    @test_throws DomainError HierarchicalGumbelCopula([1.5, 1., 0.5])
    @test_throws DomainError HierarchicalGumbelCopula([0.6, 0.4, 0.6], KendallCorrelation)
    @test_throws DomainError HierarchicalGumbelCopula([0.6, 0.4, -0.6], KendallCorrelation)

    u = zeros(3,10)
    c = HierarchicalGumbelCopula([4., 3., 2.])
    @test_throws AssertionError simulate_copula!(u, c)
  end
  @testset "simple example" begin
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, HierarchicalGumbelCopula([2., 1.8, 1.7]); rng = rng) ≈ [0.23064 0.32542 0.3123 0.76877] atol=1.0e-3

    u = zeros(1,4)
    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u, HierarchicalGumbelCopula([2., 1.8, 1.7]); rng = rng)
    @test u ≈  [0.23064 0.32542 0.3123 0.76877] atol=1.0e-3

  end
  @testset "larger example" begin

    Random.seed!(42)
    x = simulate_copula(25_000, HierarchicalGumbelCopula([4.2, 3.6, 1.1]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1:2,:] ≈ [1. 0.7619 0.72222 0.0909; 0.7619 1. 0.72222 0.0909] atol=1.0e-2
    @test tail(x[:,2], x[:,3], "r", 0.01) ≈ 2-2^(1/3.6) atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r", 0.01) ≈ 2-2^(1/1.1) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0

    # correlations
    Random.seed!(42)
    x = simulate_copula(10_000, HierarchicalGumbelCopula([0.9, 0.2], KendallCorrelation))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    c = corkendall(x)
    @test c[:,1] ≈ [1., 0.9, 0.2] atol=1.0e-2
    @test c[:,3] ≈ [0.2, 0.2, 1.] atol=1.0e-2
  end
end


@testset "test on Big Float" begin

    c1 = GumbelCopula(2, BigFloat(3.))
    c2 = GumbelCopula(3, BigFloat(4.))
    cp = NestedGumbelCopula([c1, c2], 2, BigFloat(1.5))


    Random.seed!(42)
    x = simulate_copula(100, cp)
    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α


    copula = DoubleNestedGumbelCopula([cp, cp], BigFloat(1.2))
    Random.seed!(42)
    x = simulate_copula(100, cp)
    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α


    ch = HierarchicalGumbelCopula(BigFloat.([2., 1.8, 1.7]))
    x = simulate_copula(100, ch)
    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α

end
