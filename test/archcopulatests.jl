α = 0.025

# Hepers

@testset "correlations" begin
  @testset "copula functions" begin
    @test Ccl([1., 0.5], 3.) == 0.5
    @test Cg([1., 0.5], 3.) == 0.5
  end
  @testset "Kendall cor" begin
    @test Debye(0.5, 1) ≈ 0.8819271567906056
    @test τ2θ(0.6, "frank") ≈ 7.929642284264058
    @test τ2θ(0.5, "gumbel") ≈ 2.
    @test τ2θ(1/3, "clayton") ≈ 1.
    @test τ2θ(1/4, "amh") ≈ 0.8384520912688538
    @test_throws AssertionError τ2θ(1/3, "frak")
    @test_throws DomainError τ2θ(0., "frank")
    @test frankτ2θ(0.6) ≈ 7.929642284264058
    @test AMHτ2θ(1/4) ≈ 0.8384520912688538
    @test AMHτ2θ(0.) ≈ 0. atol=1.0e-6
    @test AMHτ2θ(0.28) ≈ 0.9999
    @test AMHτ2θ(-2/11) ≈ -0.9999
  end
  @testset "Spearman cor" begin
    @test dilog(0.5) ≈ 0.5822405264650125
    @test claytonρ2θ(1/3) ≈ 0.58754 atol=1.0e-5
    @test claytonρ2θ(0.01) ≈ 0. atol=1.0e-1
    @test gumbelρ2θ(0.5) ≈ 1.5410704204332681
    @test gumbelρ2θ(0.0001) == 1.
    @test ρ2θ(1/3, "frank") ≈ 2.116497 atol=1.0e-5
    @test ρ2θ(0.2, "amh") ≈ 0.5168580913147318
    @test_throws AssertionError ρ2θ(1/3, "frak")
    @test_throws DomainError ρ2θ(0., "frank")
    @test frankρ2θ(1/3) ≈ 2.116497 atol=1.0e-5
    @test AMHρ2θ(0.2) ≈ 0.5168580913147318
    @test claytonθ2ρ(3.) ≈ 0.78645 atol=1.0e-4
    @test claytonθ2ρ(0.001) ≈ 0. atol=1.0e-2
    @test gumbelθ2ρ(3.) ≈ 0.8489 atol=1.0e-4
    @test AMHρ2θ(0.) ≈ 0. atol=1.0e-4
    @test AMHρ2θ(0.49) ≈ 1 atol=1.0e-4
    @test AMHρ2θ(-0.273) ≈ -1 atol=1.0e-4
  end
  @testset "negative cor" begin
    @test ρ2θ(-0.2246, "amh") ≈ -0.8 atol=1.0e-3
    @test τ2θ(-0.1505, "amh") ≈ -0.8 atol=1.0e-3
    @test ρ2θ(-0.5572, "frank") ≈ -4. atol=1.0e-3
    @test τ2θ(-0.3881, "frank") ≈ -4. atol=1.0e-3
    @test τ2θ(-1/3, "clayton") ≈ -.5 atol=1.0e-5
    @test ρ2θ(-0.4668, "clayton") ≈ -.5 atol=1.0e-3
  end
end
@testset "Archimedean copulas axiliary functions" begin
  c = arch_gen("clayton", [0.2 0.4 0.8; 0.2 0.8 0.6; 0.3 0.9 0.6], 1.; rng = Random.GLOBAL_RNG)
  @test c ≈ [0.5 0.637217; 0.362783 0.804163; 0.432159 0.896872] atol=1.0e-5
  @test useτ(0.5, "clayton") == 2.
  @test useρ(0.75, "gumbel") ≈ 2.285220798876495
  @test getθ4arch(0.5, "gumbel", SpearmanCorrelation) ≈ 1.541070420842913
  @test getθ4arch(0.5, "gumbel", KendallCorrelation) ≈ 2.0
end

@testset "Axiliary function exceptions" begin
  @test_throws DomainError testθ(0.5, "gumbel")
  @test_throws DomainError useρ(0.6, "amh")
  @test_throws DomainError useτ(0.45, "amh")
end

@testset "Gumbel copula" begin
  @testset "exceptions" begin
    @test_throws DomainError Gumbel_cop(3, 0.3)
    @test_throws DomainError Gumbel_cop(3, 1.1, KendallCorrelation)
    @test_throws DomainError Gumbel_cop(3, 1.1, SpearmanCorrelation)
    @test_throws DomainError Gumbel_cop(1, 1.1)
    @test_throws DomainError Gumbel_cop(1, 1.1, KendallCorrelation)

    @test_throws DomainError Gumbel_cop_rev(3, 0.3)
    @test_throws DomainError Gumbel_cop_rev(3, 1.1, KendallCorrelation)
    @test_throws DomainError Gumbel_cop_rev(3, 1.1, SpearmanCorrelation)
    @test_throws DomainError Gumbel_cop_rev(1, 1.1)
    @test_throws DomainError Gumbel_cop_rev(1, 1.1, KendallCorrelation)

    u = zeros(3,3)
    @test_throws AssertionError simulate_copula!(u, Gumbel_cop(2, 2.))
    @test_throws AssertionError simulate_copula!(u, Gumbel_cop_rev(2, 2.))
  end

  @testset "small example" begin
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, Gumbel_cop(2, 2.)) ≈ [0.481781 0.754482] atol=1.0e-5
    else
      @test simulate_copula(1, Gumbel_cop(2, 2.)) ≈ [0.189094 0.8759084] atol=1.0e-5
    end
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, Gumbel_cop_rev(2, 2.)) ≈ [0.518219 0.245518] atol=1.0e-5
    else
      @test simulate_copula(1, Gumbel_cop_rev(2, 2.)) ≈ [0.810905 0.1240915] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    simulate_copula!(u, Gumbel_cop(2, 2.))
    if VERSION <= v"1.7"
      @test u ≈ [0.481781  0.754482] atol=1.0e-5
    else
      @test u ≈ [0.189094 0.8759084] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    simulate_copula!(u, Gumbel_cop_rev(2, 2.))
    if VERSION <= v"1.7"
      @test u ≈ [0.518219  0.245518] atol=1.0e-5
    else
      @test u ≈ [0.810905 0.1240915] atol=1.0e-5
    end

    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, Gumbel_cop(2, 2.); rng = rng) ≈ [0.7622966 0.8124318] atol=1.0e-5

    Random.seed!(43)
    rng = StableRNG(123)
    u = zeros(1,2)
    simulate_copula!(u, Gumbel_cop(2, 2.); rng = rng)
    @test u ≈ [0.7622966 0.8124318] atol=1.0e-5
  end

  @testset "tests on larger data" begin
    Random.seed!(1234)
    x = simulate_copula(100_000, Gumbel_cop(3, 2.))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.
    @test corkendall(x) ≈ [1. 1/2 1/2; 1/2 1. 1/2; 1/2 1/2 1.] atol=1.0e-2

    Random.seed!(43)
    x = simulate_copula(50000, Gumbel_cop_rev(2, 1.5))
    @test tail(x[:,1], x[:,2], "l") ≈ 2-2^(1/1.5) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0.
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    Random.seed!(43)
    x = simulate_copula(50000, Gumbel_cop(2, 0.5, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(50000, Gumbel_cop(2, 0.5, SpearmanCorrelation))
    @test corspearman(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(50000, Gumbel_cop_rev(2, 0.5, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
  end
end
@testset "Clayton copula" begin
  @testset "exceptions" begin
    @test_throws DomainError Clayton_cop(3, -0.3)
    @test_throws DomainError Clayton_cop(2, -2.3)
    @test_throws DomainError Clayton_cop(3, 1.1, KendallCorrelation)
    @test_throws DomainError Clayton_cop(3, 1.1, SpearmanCorrelation)
    @test_throws DomainError Clayton_cop(1, 1.1)
    @test_throws DomainError Clayton_cop(1, 1.1, KendallCorrelation)

    @test_throws DomainError Clayton_cop_rev(3, -0.3)
    @test_throws DomainError Clayton_cop_rev(2, -2.3)
    @test_throws DomainError Clayton_cop_rev(3, 1.1, KendallCorrelation)
    @test_throws DomainError Clayton_cop_rev(3, 1.1, SpearmanCorrelation)
    @test_throws DomainError Clayton_cop_rev(1, 1.1)
    @test_throws DomainError Clayton_cop_rev(1, 1.1, KendallCorrelation)

    u = zeros(3,3)
    @test_throws AssertionError simulate_copula!(u, Clayton_cop(2, 2.))
    @test_throws AssertionError simulate_copula!(u, Clayton_cop_rev(2, 2.))
  end
  @testset "small example" begin
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, Clayton_cop(2, 2.)) ≈ [0.652812  0.912719] atol=1.0e-5
      @test simulate_copula(1, Clayton_cop(2, -0.5)) ≈ [0.924876  0.185707] atol=1.0e-5
    else
      @test simulate_copula(1, Clayton_cop(2, 2.)) ≈ [0.2531841 0.9567753] atol=1.0e-5
      @test simulate_copula(1, Clayton_cop(2, -0.5)) ≈ [0.428396 0.4814575]  atol=1.0e-5
    end
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, Clayton_cop_rev(2, 2.)) ≈ [0.347188  0.087281] atol=1.0e-5
      @test simulate_copula(1, Clayton_cop_rev(2, -.5)) ≈ 1. .- [0.924876  0.185707] atol=1.0e-5
    else
      @test simulate_copula(1, Clayton_cop_rev(2, 2.)) ≈ [0.746815 0.0432246] atol=1.0e-5
      @test simulate_copula(1, Clayton_cop_rev(2, -.5)) ≈ [0.57160 0.5185424] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    u1 = zeros(1,2)
    simulate_copula!(u, Clayton_cop(2, 2.))
    simulate_copula!(u1, Clayton_cop(2, -0.5))
    if VERSION <= v"1.7"
      @test u ≈ [0.652812  0.912719] atol=1.0e-5
      @test u1 ≈ [0.924876  0.185707] atol=1.0e-5
    else
      @test u ≈  [0.253184 0.9567753] atol=1.0e-5
      @test u1 ≈ [0.428396 0.4814575] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    u1 = zeros(1,2)
    simulate_copula!(u, Clayton_cop_rev(2, 2.))
    simulate_copula!(u1, Clayton_cop_rev(2, -0.5))
    if VERSION <= v"1.7"
      @test u ≈ 1 .- [0.652812  0.912719] atol=1.0e-5
      @test u1 ≈ 1 .- [0.924876  0.185707] atol=1.0e-5
    else
      @test u ≈ 1 .- [0.253184 0.9567753] atol=1.0e-5
      @test u1 ≈ 1 .- [0.428396 0.4814575] atol=1.0e-5
    end
  end
  @testset "test on larger data" begin
    Random.seed!(43)
    x = simulate_copula(50000, Clayton_cop(3, 1.))

    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    @test corkendall(x) ≈ [1. 1/3 1/3; 1/3 1. 1/3; 1/3 1/3 1.] atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(50000, Clayton_cop(2, 0.5, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(50000, Clayton_cop(2, -0.9))
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-2

    Random.seed!(43)
    x = simulate_copula(50000, Clayton_cop_rev(2, 0.5, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2

    Random.seed!(43)
    x = simulate_copula(50000, Clayton_cop_rev(2, -0.9))
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-2

  end
end

@testset "Ali-Mikhail-Haq copula" begin
  @testset "exceptions" begin
    @test_throws DomainError AMH_cop(3, -0.3)
    @test_throws DomainError AMH_cop(2, -1.2)
    @test_throws DomainError AMH_cop(3, 0.5, KendallCorrelation)
    @test_throws DomainError AMH_cop(3, 0.6, SpearmanCorrelation)
    @test_throws DomainError AMH_cop(1, 1.1)
    @test_throws DomainError AMH_cop(1, 1.1, KendallCorrelation)

    @test_throws DomainError AMH_cop_rev(3, -0.3)
    @test_throws DomainError AMH_cop_rev(2, -1.2)
    @test_throws DomainError AMH_cop_rev(3, 0.5, KendallCorrelation)
    @test_throws DomainError AMH_cop_rev(3, 0.55, SpearmanCorrelation)
    @test_throws DomainError AMH_cop_rev(1, 1.1)
    @test_throws DomainError AMH_cop_rev(1, 1.1, KendallCorrelation)

    u = zeros(3,3)
    @test_throws AssertionError simulate_copula!(u, AMH_cop(2, .5))
    @test_throws AssertionError simulate_copula!(u, AMH_cop_rev(2, .5))
  end
  @testset "small example" begin
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, AMH_cop(2, 0.5)) ≈ [0.483939  0.883911] atol=1.0e-5
      @test simulate_copula(1, AMH_cop(2, -0.5)) ≈ [0.924876  0.320496] atol=1.0e-5
    else
      @test simulate_copula(1, AMH_cop(2, 0.5)) ≈ [0.087839 0.9773157]  atol=1.0e-5
      @test simulate_copula(1, AMH_cop(2, -0.5)) ≈ [0.428396 0.5532944]  atol=1.0e-5
    end
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, AMH_cop_rev(2, 0.5)) ≈ 1 .- [0.483939  0.883911] atol=1.0e-5
      @test simulate_copula(1, AMH_cop_rev(2, -0.5)) ≈ 1 .- [0.924876  0.320496] atol=1.0e-5
    else
      @test simulate_copula(1, AMH_cop_rev(2, 0.5)) ≈ 1 .- [0.087839 0.9773157] atol=1.0e-5
      @test simulate_copula(1, AMH_cop_rev(2, -0.5)) ≈ 1 .- [0.428396 0.5532944] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    u1 = zeros(1,2)
    simulate_copula!(u, AMH_cop(2, 0.5))
    simulate_copula!(u1, AMH_cop(2, -0.5))
    if VERSION <= v"1.7"
      @test u ≈ [0.483939  0.883911] atol=1.0e-5
      @test u1 ≈ [0.924876  0.320496] atol=1.0e-5
    else
      @test u ≈ [0.087839 0.9773157] atol=1.0e-5
      @test u1 ≈ [0.428396 0.5532944] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    u1 = zeros(1,2)
    simulate_copula!(u, AMH_cop_rev(2, 0.5))
    simulate_copula!(u1, AMH_cop_rev(2, -0.5))
    if VERSION <= v"1.7"
      @test u ≈ 1 .- [0.483939  0.883911] atol=1.0e-5
      @test u1 ≈ 1 .- [0.924876  0.320496] atol=1.0e-5
    else
      @test u ≈ 1 .- [0.087839 0.9773157] atol=1.0e-5
      @test u1 ≈ 1 .- [0.428396 0.5532944] atol=1.0e-5
    end
  end
  @testset "test on larger data" begin

    Random.seed!(43)
    x = simulate_copula(100_000, AMH_cop(3, 0.8))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    @test corkendall(x)[1:2, 1:2] ≈ [1. 0.233; 0.233 1.] atol=1.0e-2
    x = simulate_copula(100_000, AMH_cop(2, 0.25, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.25; 0.25 1.] atol=1.0e-3

    Random.seed!(43)
    x = simulate_copula(100000, AMH_cop_rev(3, 0.8))

    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

    Random.seed!(43)
    x = simulate_copula(100000, AMH_cop_rev(2, -0.4))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

    Random.seed!(43)
    x = simulate_copula(100000, AMH_cop_rev(2, 0.2, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.2; 0.2 1.] atol=1.0e-2
  end
end

@testset "Frank copula" begin
  @testset "exceptions" begin
    @test_throws DomainError Frank_cop(3, -0.3)
    @test_throws DomainError Frank_cop(2, 0.)
    @test_throws DomainError Frank_cop(3, 1.1, KendallCorrelation)
    @test_throws DomainError Frank_cop(3, 1.1, SpearmanCorrelation)
    @test_throws DomainError Frank_cop(1, 1.1)
    @test_throws DomainError Frank_cop(1, 1.1, KendallCorrelation)

    u = zeros(3,3)
    @test_throws AssertionError simulate_copula!(u, Frank_cop(2, .5))
  end
  @testset "small example" begin
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, Frank_cop(2, 2.)) ≈ [0.565546  0.897293] atol=1.0e-5
      @test simulate_copula(1, Frank_cop(2, -2.)) ≈ [0.924876  0.242893] atol=1.0e-5
    else
      @test simulate_copula(1, Frank_cop(2, 2.)) ≈ [0.565546  0.897293] atol=1.0e-5
      @test simulate_copula(1, Frank_cop(2, -2.)) ≈ [0.924876  0.242893] atol=1.0e-5
    end

    Random.seed!(43)
    u = zeros(1,2)
    u1 = zeros(1,2)
    simulate_copula!(u, Frank_cop(2, 2.))
    simulate_copula!(u1, Frank_cop(2, -2.))
    if VERSION <= v"1.7"
      @test u ≈ [0.565546  0.897293] atol=1.0e-5
      @test u1 ≈ [0.924876  0.242893] atol=1.0e-5
    else
      @test u ≈ [0.565546  0.897293] atol=1.0e-5
      @test u1 ≈ [0.924876  0.242893] atol=1.0e-5
    end

  end
  @testset "test on larger data" begin
    Random.seed!(43)
    x = simulate_copula(300000, Frank_cop(3, 0.8))

    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0
    Random.seed!(43)
    x = simulate_copula(100000, Frank_cop(2, 0.2, KendallCorrelation))
    @test corkendall(x) ≈ [1. 0.2; 0.2 1.] atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(100000, Frank_cop(2, -2.))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  end
end

@testset "tests on Big Float" begin
  Random.seed!(1234)
  θ = BigFloat(2.)
  x = simulate_copula(1000, Gumbel_cop(3, θ))

  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α


  Random.seed!(1234)
  θ = BigFloat(-.5)
  x = simulate_copula(1000, Clayton_cop(2, θ))

  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

  Random.seed!(1234)
  θ = BigFloat(2.5)
  if false
    x = simulate_copula(1000, Clayton_cop(3, θ))

    @test typeof(x) == Array{BigFloat,2}
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  end

  Random.seed!(1234)
  θ = BigFloat(2.)
  x = simulate_copula(1000, Frank_cop(2, θ))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

  Random.seed!(1234)
  θ = BigFloat(.5)
  x = simulate_copula(10, AMH_cop(3, θ))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

  Random.seed!(1234)
  θ = BigFloat(-.3)
  x = simulate_copula(10, AMH_cop(2, θ))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

end
