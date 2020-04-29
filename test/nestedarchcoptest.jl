α = 0.025

@testset "nested archimedean copulas helpers 4 higher correlations" begin
  Random.seed!(43)
  u = nestedcopulag("clayton", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
  @test u ≈ [0.1532819 0.1824212 0.3742276 0.4076627; 0.690350 0.740927 0.254841 0.279192] atol=1.0e-5
  Random.seed!(42)
  x = nestedcopulag("gumbel", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
  @test x ≈ [0.624812  0.674897  0.451637  0.483973 ;  0.800605  0.825022  0.907411  0.915277] atol=1.0e-5
end

@testset "nested Clayton copula" begin
  @testset "exceptions" begin
    a = Clayton_cop(2, 1.)
    b = Clayton_cop(2, 2.)
    c = Clayton_cop(2, 10.)
    d = Clayton_cop(2, 20.)
    @test_throws DomainError Nested_Clayton_cop([a,b], 0, 1.5)
    @test_throws DomainError Nested_Clayton_cop([a,b], -1, 0.5)
    #@test_warn "θ << ϕ, marginals may not be uniform" Nested_Clayton_cop([c,d], 0, 0.05)
  end
  @testset "small example" begin
      c1 = Clayton_cop(2, 2.)
      c2 = Clayton_cop(2, 3.)
      cp = Nested_Clayton_cop([c1, c2], 1, 1.1)

      Random.seed!(43)
      @test simulate_copula(1, cp) ≈ [0.514118  0.84089  0.870106  0.906233  0.739349] atol=1.0e-5
  end
  @testset "large example on data" begin
      c1 = Clayton_cop(2, 3.)
      c2 = Clayton_cop(3, 4.)
      cp = Nested_Clayton_cop([c1, c2], 2, 1.5)
      # test old dispatching
      Random.seed!(42)
      x1 = nestedarchcopulagen(1000, [2, 3],  [3., 4.], 1.5, "clayton", 2)
      Random.seed!(42)
      x2 = simulate_copula(1000, cp)
      @test norm(x1 -x2) ≈ 0.

      Random.seed!(42)
      x = simulate_copula(80000, cp)
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
      x = simulate_copula(75000, cp)
      @test corkendall(x)[:,1] ≈ [1, 0.7, 0.3] atol=1.0e-2
    end
end

@testset "nested Ali-Mikhail-Haq copula" begin
  @testset "exceptions" begin
    a = AMH_cop(2, .8)
    b = AMH_cop(2, .3)
    @test_throws DomainError Nested_AMH_cop([a,b], 0, 0.5)
    @test_throws DomainError Nested_AMH_cop([a,b], -1, 0.1)
  end
  @testset "small example" begin
      c1 = AMH_cop(2, .8)
      c2 = AMH_cop(2, .9)
      cp = Nested_AMH_cop([c1, c2], 1, 0.5)
      Random.seed!(43)
      @test simulate_copula(1, cp) ≈ [0.587322  0.910074  0.931225  0.953353  0.665769] atol=1.0e-5
  end
  @testset "large example" begin
      c1 = AMH_cop(3, .8)
      c2 = AMH_cop(2, .7)
      cp = Nested_AMH_cop([c1, c2], 2, 0.5)
      # test old dispatching
      Random.seed!(43)
      x2 = simulate_copula(1000, cp)
      Random.seed!(43)
      x1 = nestedarchcopulagen(1000, [3, 2], [0.8, 0.7], 0.5, "amh", 2)
      @test norm(x1 -x2) ≈ 0.

      Random.seed!(44)
      x = simulate_copula(100_000, cp)
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
      x = simulate_copula(80000, cp)
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
  end
  @testset "small data set" begin
    Random.seed!(43)
    a = Frank_cop(2, 2.)
    b = Frank_cop(2, 3.)
    cp = Nested_Frank_cop([a,b], 1, 1.1)
    @test simulate_copula(1, cp) ≈ [0.599183  0.908848  0.950577  0.966366  0.692735] atol=1.0e-5
  end
  @testset "large data set" begin
    # test old dispatching
    a = Frank_cop(3, 8.)
    b = Frank_cop(2, 10.)
    cp = Nested_Frank_cop([a,b], 2, 2.)
    Random.seed!(44)
    x3 = simulate_copula(1000, cp)

    Random.seed!(44)
    x1 = nestedarchcopulagen(1000, [3, 2],  [8., 10.], 2., "frank", 2)
    @test norm(x1 -x3) ≈ 0.

    Random.seed!(43)
    x = simulate_copula(200_000, cp)
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
    x = simulate_copula(80000, cp)
    @test corkendall(x)[:,1] ≈ [1, 0.6, 0.2] atol=1.0e-2
  end
end

@testset "single nested Gumbel" begin
  @testset "exceptios" begin
    a = Gumbel_cop(2, 2.)
    b = Gumbel_cop(2, 1.5)
    @test_throws DomainError Nested_Gumbel_cop([a,b], 0, 1.7)
    @test_throws DomainError Nested_Gumbel_cop([a,b], -1, 1.1)
  end
  @testset "test on small data" begin
    a = Gumbel_cop(2, 2.)
    b = Gumbel_cop(2, 3.)
    cp = Nested_Gumbel_cop([a,b], 1, 1.1)
    Random.seed!(43)
    @test simulate_copula(1, cp) ≈ [0.387085  0.693399  0.94718  0.953776  0.583379] atol=1.0e-5
    Random.seed!(43)
    n = nestedcopulag("gumbel", [[1,2], [3,4]], [2., 3.], 1.1,  [0.1 0.2 0.3 0.4 0.5; 0.1 0.2 0.3 0.4 0.5])
    @test n ≈ [0.00963842 0.0206319 0.556597 0.585703; 0.0772418 0.117543 0.838879 0.8518] atol=1.0e-5
  end
  @testset "test on larger data" begin
    a = Gumbel_cop(2, 4.2)
    b = Gumbel_cop(2, 6.1)
    cp = Nested_Gumbel_cop([a,b], 1, 2.1)
    # test old dispatching
    Random.seed!(44)
    x1 = nestedarchcopulagen(1000, [2,2], [4.2, 6.1], 2.1, "gumbel", 1)
    Random.seed!(44)
    x3 = simulate_copula(1000, cp)
    @test norm(x1 -x3) ≈ 0.

    Random.seed!(44)
    x = simulate_copula(1_000_000, cp)
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
    x = simulate_copula(80000, cp)
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
    @test simulate_copula(1, copula) ≈ [0.598555  0.671584  0.8403  0.846844  0.634609  0.686927  0.693906  0.651968  0.670812] atol=1.0e-5
  end
  @testset "large data" begin
    a = Gumbel_cop(2, 4.1)
    b = Gumbel_cop(2, 3.8)
    cp = Nested_Gumbel_cop([a,b], 0, 1.9)

    a1 = Gumbel_cop(2, 5.1)
    b1 = Gumbel_cop(2, 6.1)
    cp1 = Nested_Gumbel_cop([a1, b1], 0, 2.4)
    cgp = Double_Nested_Gumbel_cop([cp, cp1], 1.2)

    # test dispatching
    Random.seed!(43)
    x1 = nestedarchcopulagen(1000, [[2,2], [2,2]], [[4.1, 3.8],[5.1, 6.1]], [1.9, 2.4], 1.2, "gumbel")
    Random.seed!(43)
    x3 = simulate_copula(1000, cgp)
    @test norm(x3 - x1) ≈ 0.

    Random.seed!(43)
    x = simulate_copula(200000, cgp)
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

    x = simulate_copula(250000, cgp)
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
  end
  @testset "simple example" begin
    Random.seed!(43)
    @test simulate_copula(1, Hierarchical_Gumbel_cop([2., 1.8, 1.7])) ≈ [0.117637  0.437958  0.150743  0.0954366] atol=1.0e-5
  end
  @testset "larger example" begin
    # test old dispatching
    Random.seed!(42)
    x1 = nestedarchcopulagen(1000, [4.2, 3.6, 1.1], "gumbel")
    Random.seed!(42)
    x2 = simulate_copula(1000, Hierarchical_Gumbel_cop([4.2, 3.6, 1.1]))
    @test norm(x2 - x1) ≈ 0.

    Random.seed!(42)
    x = simulate_copula(500000, Hierarchical_Gumbel_cop([4.2, 3.6, 1.1]))
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
    x = simulate_copula(500000, Hierarchical_Gumbel_cop([0.9, 0.2], KendallCorrelation))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    c = corkendall(x)
    @test c[:,1] ≈ [1., 0.9, 0.2] atol=1.0e-2
    @test c[:,3] ≈ [0.2, 0.2, 1.] atol=1.0e-2
  end
end
