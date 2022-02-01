α = 0.025

@testset "nested archimedean copulas helpers 4 higher correlations" begin
  Random.seed!(43)
  u = nestedcopulag("clayton", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6]; rng = Random.GLOBAL_RNG)
  if VERSION <= v"1.7"
      @test u ≈ [0.1532 0.1824 0.3742 0.4076; 0.6903 0.7409 0.2548 0.2791] atol=1.0e-3
  else
      @test u ≈ [0.3491 0.40705 0.6121 0.6551; 0.3630 0.41073 0.6927 0.7349] atol=1.0e-3
  end
  Random.seed!(42)
  x = nestedcopulag("gumbel", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6]; rng = Random.GLOBAL_RNG)
  if VERSION <= v"1.7"
      @test x ≈ [0.6248 0.6748 0.4516 0.4839; 0.8006 0.8250 0.9074 0.9152] atol=1.0e-3
  else
      @test x ≈ [0.3974 0.4623 0.2064 0.2368; 0.2826 0.3352 0.3489 0.38313] atol=1.0e-3
  end
end

@testset "nested Clayton copula" begin
  @testset "exceptions" begin
    a = Clayton_cop(2, 1.)
    b = Clayton_cop(2, 2.)
    c = Clayton_cop(2, 10.)
    d = Clayton_cop(2, 20.)

    @test_throws DomainError Nested_Clayton_cop([a,b], 0, 1.5)
    @test_throws DomainError Nested_Clayton_cop([a,b], -1, 0.5)

    u = zeros(3,5)
    cp = Nested_Clayton_cop([a,b], 0, 0.5)
    @test_throws AssertionError simulate_copula!(u, cp)
    #@test_warn "θ << ϕ, marginals may not be uniform" Nested_Clayton_cop([c,d], 0, 0.05)
  end
  @testset "small example" begin
      c1 = Clayton_cop(2, 2.)
      c2 = Clayton_cop(2, 3.)
      cp = Nested_Clayton_cop([c1, c2], 1, 1.1)

      Random.seed!(43)
      if VERSION <= v"1.7"
          @test simulate_copula(1, cp)[:,1:2] ≈ [0.514118  0.84089] atol=1.0e-5
      else
          @test simulate_copula(1, cp)[:,1:2] ≈ [0.402091 0.98398] atol=1.0e-5
      end

      u = zeros(1,5)
      Random.seed!(43)
      simulate_copula!(u, cp)
      if VERSION <= v"1.7"
          @test u ≈ [0.514118  0.84089  0.870106  0.906233  0.739349] atol=1.0e-5
      else
          @test u[:,1:2] ≈ [0.402091 0.98398] atol=1.0e-5
      end

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
      c1 = Clayton_cop(2, 3.)
      c2 = Clayton_cop(3, 4.)
      cp = Nested_Clayton_cop([c1, c2], 2, 1.5)


      Random.seed!(42)
      x = simulate_copula(30_000, cp)
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

      #test on correlations
      c1 = Clayton_cop(2, .7, KendallCorrelation)
      cp = Nested_Clayton_cop([c1], 1, 0.3, KendallCorrelation)
      x = simulate_copula(30_000, cp)
      @test corkendall(x)[:,1] ≈ [1, 0.7, 0.3] atol=1.0e-2
    end
end

@testset "nested Ali-Mikhail-Haq copula" begin
  @testset "exceptions" begin
    a = AMH_cop(2, .8)
    b = AMH_cop(2, .3)
    @test_throws DomainError Nested_AMH_cop([a,b], 0, 0.5)
    @test_throws DomainError Nested_AMH_cop([a,b], -1, 0.1)

    u = zeros(5, 3)
    cp = Nested_AMH_cop([a,b], 0, 0.1)
    @test_throws AssertionError simulate_copula!(u, cp)
  end
  @testset "small example" begin
      c1 = AMH_cop(2, .8)
      c2 = AMH_cop(2, .9)
      cp = Nested_AMH_cop([c1, c2], 1, 0.5)
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
      c1 = AMH_cop(3, .8)
      c2 = AMH_cop(2, .7)
      cp = Nested_AMH_cop([c1, c2], 2, 0.5)


      Random.seed!(44)
      x = simulate_copula(30_000, cp)
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

      #test on correlations
      c1 = AMH_cop(2, .2, KendallCorrelation)
      cp = Nested_AMH_cop([c1], 1, 0.1, KendallCorrelation)
      x = simulate_copula(10_000, cp)
      @test corkendall(x)[:,1] ≈ [1, 0.2, 0.1] atol=1.0e-2
  end
end

@testset "nested Frank copula" begin
  @testset "exceptions" begin
    a = Frank_cop(2, 2.)
    b = Frank_cop(2, -1.)
    c = Frank_cop(2, 1.)
    @test_throws DomainError Nested_Frank_cop([a,b], 0, 0.5)
    @test_throws DomainError Nested_Frank_cop([a,c], -1, 0.1)

    u = zeros(5, 7)
    cp = Nested_Frank_cop([a,c], 0, 0.1)
    @test_throws AssertionError simulate_copula!(u, cp)
  end
  @testset "small data set" begin
    a = Frank_cop(2, 2.)
    b = Frank_cop(2, 3.)
    cp = Nested_Frank_cop([a,b], 1, 1.1)

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

    a = Frank_cop(3, 8.)
    b = Frank_cop(2, 10.)
    cp = Nested_Frank_cop([a,b], 2, 2.)


    Random.seed!(43)
    x = simulate_copula(40_000, cp)
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
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

    # correlation tests
    c1 = Frank_cop(2, .6, KendallCorrelation)
    cp = Nested_Frank_cop([c1], 1, 0.2, KendallCorrelation)
    x = simulate_copula(10_000, cp)
    @test corkendall(x)[:,1] ≈ [1, 0.6, 0.2] atol=1.0e-2
  end
end

@testset "single nested Gumbel" begin
  @testset "exceptios" begin
    a = Gumbel_cop(2, 2.)
    b = Gumbel_cop(2, 1.5)
    @test_throws DomainError Nested_Gumbel_cop([a,b], 0, 1.7)
    @test_throws DomainError Nested_Gumbel_cop([a,b], -1, 1.1)

    u = zeros(5, 7)
    cp = Nested_Gumbel_cop([a,b], 1, 1.1)
    @test_throws AssertionError simulate_copula!(u, cp)
  end
  @testset "test on small data" begin
    a = Gumbel_cop(2, 2.)
    b = Gumbel_cop(2, 3.)
    cp = Nested_Gumbel_cop([a,b], 1, 1.1)
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
    a = Gumbel_cop(2, 4.2)
    b = Gumbel_cop(2, 6.1)
    cp = Nested_Gumbel_cop([a,b], 1, 2.1)

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
    c1 = Gumbel_cop(2, .8, KendallCorrelation)
    cp = Nested_Gumbel_cop([c1], 1, 0.2, KendallCorrelation)
    x = simulate_copula(10_000, cp)
    @test corkendall(x)[:,1] ≈ [1, 0.8, 0.2] atol=1.0e-2
  end
end

@testset "double nested Gumbel copula" begin
  @testset "exceptions" begin
    a = Gumbel_cop(2, 4.2)
    b = Gumbel_cop(2, 6.1)
    cp = Nested_Gumbel_cop([a,b], 1, 2.1)

    a1 = Gumbel_cop(2, 4.2)
    b1 = Gumbel_cop(2, 5.1)
    cp1 = Nested_Gumbel_cop([a1,b1], 0, 3.1)

    @test_throws DomainError Double_Nested_Gumbel_cop([cp, cp1], 2.2)
    @test_throws DomainError Double_Nested_Gumbel_cop([cp, cp1], 0.9, KendallCorrelation)
    copula = Double_Nested_Gumbel_cop([cp, cp1], 1.2)
    u = rand(3,20)
    @test_throws AssertionError simulate_copula!(u, copula)
  end
  @testset "small data" begin
    Random.seed!(43)
    a = Gumbel_cop(2, 5.)
    b = Gumbel_cop(2, 6.)
    p1 = Nested_Gumbel_cop([a,b], 1, 2.)

    c = Gumbel_cop(2, 5.5)
    p2 = Nested_Gumbel_cop([c], 2, 2.1)

    copula = Double_Nested_Gumbel_cop([p1, p2], 1.5)
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
    a = Gumbel_cop(2, 4.1)
    b = Gumbel_cop(2, 3.8)
    cp = Nested_Gumbel_cop([a,b], 0, 1.9)

    a1 = Gumbel_cop(2, 5.1)
    b1 = Gumbel_cop(2, 6.1)
    cp1 = Nested_Gumbel_cop([a1, b1], 0, 2.4)
    cgp = Double_Nested_Gumbel_cop([cp, cp1], 1.2)


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
    a = Gumbel_cop(2, 0.8, KendallCorrelation)
    cp = Nested_Gumbel_cop([a], 1, 0.5, KendallCorrelation)

    a1 = Gumbel_cop(2, 0.7, KendallCorrelation)
    cp1 = Nested_Gumbel_cop([a1], 1, 0.4, KendallCorrelation)
    cgp = Double_Nested_Gumbel_cop([cp, cp1], 0.2, KendallCorrelation)

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
    @test_throws DomainError Hierarchical_Gumbel_cop([5., 6., 7.])
    @test_throws DomainError Hierarchical_Gumbel_cop([1.5, 1., 0.5])
    @test_throws DomainError Hierarchical_Gumbel_cop([0.6, 0.4, 0.6], KendallCorrelation)
    @test_throws DomainError Hierarchical_Gumbel_cop([0.6, 0.4, -0.6], KendallCorrelation)

    u = zeros(3,10)
    c = Hierarchical_Gumbel_cop([4., 3., 2.])
    @test_throws AssertionError simulate_copula!(u, c)
  end
  @testset "simple example" begin
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, Hierarchical_Gumbel_cop([2., 1.8, 1.7]); rng = rng) ≈ [0.23064 0.32542 0.3123 0.76877] atol=1.0e-3

    u = zeros(1,4)
    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u, Hierarchical_Gumbel_cop([2., 1.8, 1.7]); rng = rng)
    @test u ≈  [0.23064 0.32542 0.3123 0.76877] atol=1.0e-3

  end
  @testset "larger example" begin

    Random.seed!(42)
    x = simulate_copula(10_000, Hierarchical_Gumbel_cop([4.2, 3.6, 1.1]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
    @test corkendall(x)[1:2,:] ≈ [1. 0.7619 0.72222 0.0909; 0.7619 1. 0.72222 0.0909] atol=1.0e-2
    @test tail(x[:,2], x[:,3], "r", 0.01) ≈ 2-2^(1/3.6) atol=1.0e-1
    @test tail(x[:,3], x[:,4], "r", 0.01) ≈ 2-2^(1/1.1) atol=1.0e-2
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0

    # correlations
    Random.seed!(42)
    x = simulate_copula(10_000, Hierarchical_Gumbel_cop([0.9, 0.2], KendallCorrelation))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    c = corkendall(x)
    @test c[:,1] ≈ [1., 0.9, 0.2] atol=1.0e-2
    @test c[:,3] ≈ [0.2, 0.2, 1.] atol=1.0e-2
  end
end


@testset "test on Big Float" begin
    if false
        c1 = Clayton_cop(2, BigFloat(3.))
        c2 = Clayton_cop(3, BigFloat(4.))
        cp = Nested_Clayton_cop([c1, c2], 2, BigFloat(1.5))

        Random.seed!(42)
        x = simulate_copula(10, cp)
        @test typeof(x) == Array{BigFloat,2}
        @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
        @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
        @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    end

    c1 = Gumbel_cop(2, BigFloat(3.))
    c2 = Gumbel_cop(3, BigFloat(4.))
    cp = Nested_Gumbel_cop([c1, c2], 2, BigFloat(1.5))


    Random.seed!(42)
    x = simulate_copula(100, cp)
    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α


    copula = Double_Nested_Gumbel_cop([cp, cp], BigFloat(1.2))
    Random.seed!(42)
    x = simulate_copula(100, cp)
    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α


    ch = Hierarchical_Gumbel_cop(BigFloat.([2., 1.8, 1.7]))
    x = simulate_copula(100, ch)
    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α


    if false
        c1 = AMH_cop(2, BigFloat(0.3))
        c2 = AMH_cop(3, BigFloat(0.5))
        cp = Nested_AMH_cop([c1, c2], 2, BigFloat(.2))

        Random.seed!(42)
        x = simulate_copula(100, cp)
        println(typeof(x) == Array{BigFloat,2})
        @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
        @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
        @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    end
    if false

        c1 = Frank_cop(2, BigFloat(1.1))
        #c2 = Frank_cop(3, BigFloat(2.))
        cp = Nested_Frank_cop([c1], 1, BigFloat(.9))

        Random.seed!(42)
        x = simulate_copula(2, cp)
        @test typeof(x) == Array{BigFloat,2}
        #@test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
        #@test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
        #@test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α

    end
end
