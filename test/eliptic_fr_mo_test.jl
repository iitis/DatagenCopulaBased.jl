α = 0.025

@testset "gaussian copula" begin
  @testset "exceptions" begin
    @test_throws DomainError GaussianCopula([1. 0.; .1 1.])
    @test_throws DomainError GaussianCopula([1. 2.; .2 1.])
    @test_throws DomainError GaussianCopula([2. 1.; 1. 1.])
  end
  @testset "examples on data" begin
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, GaussianCopula([1. 0.; 0. 1.]); rng = rng) ≈ [0.44953 0.74757] atol=1.0e-5

    Random.seed!(43)
    x = simulate_copula(350000, GaussianCopula([1. 0.5; 0.5 1.]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0
  end

end
@testset "t-student copula" begin
  @testset "exceptions" begin
    @test_throws DomainError StudentCopula([1. 0.; .1 1.], 2)
    @test_throws DomainError StudentCopula([1. 2.; .2 1.], 2)
    @test_throws DomainError StudentCopula([2. 1.; 1. 1.], 2)
    @test_throws DomainError StudentCopula([1. 0.; 0. 1.], 0)
  end

  @testset "example on data" begin
    Random.seed!(43)
    rng = StableRNG(123)
    v = simulate_copula(1, StudentCopula([1. 0.; 0. 1.], 1); rng = rng) 
    @test size(v) == (1, 2)

    ν = 10
    rho = 0.5
    λ = 2*pdf(TDist(ν+1), -sqrt.((ν+1)*(1-rho)/(1+rho)))
    Random.seed!(43)
    xt = simulate_copula(350000, StudentCopula([1. rho; rho 1.], ν))
    @test pvalue(ExactOneSampleKSTest(xt[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xt[:,2], Uniform(0,1))) > α
    @test tail(xt[:,1], xt[:,2], "l") ≈ λ atol=1.0e-1
    @test tail(xt[:,1], xt[:,2], "r") ≈ λ atol=1.0e-1

    convertmarg!(xt, Normal)
    @test cor(xt) ≈ [1. rho; rho 1.] atol=1.0e-2
  end
end

@testset "Frechet copula" begin
  @testset "Frechet copula exceptions" begin
    @test_throws DomainError FrechetCopula(3, -0.8)
    @test_throws DomainError FrechetCopula(1, 0.8)

    @test_throws AssertionError FrechetCopula(3, 0.8, 0.1)
    @test_throws DomainError FrechetCopula(2, 0.8, 0.5)
    @test_throws DomainError FrechetCopula(2, -0.2, 0.8)
    @test_throws DomainError FrechetCopula(2, 0.8, -0.5)

    u = zeros(3, 5)
    cop = FrechetCopula(3, 0.8)
    @test_throws AssertionError simulate_copula!(u, cop)
  end
  @testset "small example" begin
    u = [0.3, 0.5, 0.7]
    frechet_el!(u, 0.3, 0.5)
    @test u == [0.3, 0.5, 0.7]
    frechet_el!(u, 0.7, 0.5)
    @test u == [0.7, 0.7, 0.7]
    u = [0.1, 0.8]
    frechet_el2!(u, 0.2, 0.6, 0.6)
    @test u ≈ [0.2, 0.8]
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, FrechetCopula(4, 1.); rng = rng) ≈ [0.042730 0.04273 0.04273 0.042730] atol=1.0e-5

    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, FrechetCopula(4, 1.); rng = rng) ≈ [0.042730 0.042730 0.04273 0.04273] atol=1.0e-5

    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, FrechetCopula(2, 0.4, 0.4); rng = rng) ≈ [0.632510 0.36748] atol=1.0e-5


    u = zeros(1,4)
    u1 = zeros(1,2)
    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u, FrechetCopula(4, 1.); rng = rng)
    @test u ≈ [0.042730 0.042730 0.04273 0.04273] atol=1.0e-5

    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u1, FrechetCopula(2, 0.4, 0.4); rng = rng)
    @test u1 ≈ [0.63251 0.36748] atol=1.0e-5

  end
  @testset "examples on larger data" begin
    Random.seed!(43)
    x = simulate_copula(350000, FrechetCopula(3, 0.3))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l") ≈ 0.3 atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r") ≈ 0.3 atol=1.0e-1
    @test corspearman(x) ≈ [1. 0.3 0.3; 0.3 1. 0.3; 0.3 0.3 1.] atol=1.0e-2


    x = simulate_copula(350000, FrechetCopula(3, 1.))
    ret = (x[:, 1] .<  0.2).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
    @test length(findall(ret))/size(x,1) ≈ 0.2 atol=1.0e-3
    ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
    @test length(findall(ret))/size(x,1) ≈ 0.3 atol=1.0e-3
    ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.9).*(x[:, 3] .<  0.5)
    @test length(findall(ret))/size(x,1) ≈ 0.5 atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(1500000, FrechetCopula(2, .0))
    @test corspearman(x) ≈ [1. 0.; 0. 1.] atol=1.0e-2

    x = simulate_copula(350000, FrechetCopula(2, 0.8, 0.1));
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test corspearman(x) ≈ [1. 0.7; 0.7 1.] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "l") ≈ 0.8 atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r") ≈ 0.8 atol=1.0e-1
  end
end


@testset "Marshall-Olkin copula" begin
  @testset "helpers and exceptions" begin
    @test_throws AssertionError MarshallOlkinCopula([1., -2., 3.])
    @test_throws AssertionError MarshallOlkinCopula([1., 3.])
    @test τ2λ([0.4], [4.2, 1.2]) ≈ [4.2, 1.2, 3.6]
    @test τ2λ([0.5, 0.6, 0.5], [1., 2., 3., 4.]) ≈ [1, 2, 3, 0, 0.5, 0, 4]
    @test moρ2τ(0.6) ≈ 0.5 atol=1.0e-2

    u = zeros(3,3)
    cop = MarshallOlkinCopula([1.,2.,3.])
    @test_throws AssertionError simulate_copula!(u, cop)
  end
  @testset "small example" begin
    Random.seed!(43)
    rng = StableRNG(123)
    @test simulate_copula(1, MarshallOlkinCopula([1., 2., 3.]); rng = rng) ≈ [0.585174 0.511807] atol=1.0e-5


    u = zeros(1,2)
    Random.seed!(43)
    rng = StableRNG(123)
    simulate_copula!(u, MarshallOlkinCopula([1., 2., 3.]); rng=rng)
    @test u ≈ [0.585174 0.511807] atol=1.0e-5


    m = [0.252982 0.201189;  0.464758 0.409039; 0.585662 0.5357]
    @test mocopula([0.2 0.3 0.4; 0.3 0.4 0.6; 0.4 0.5 0.7], 2, [1., 1.5, 2.]) ≈ m atol=1.0e-4

    s = collect(combinations(1:2))
    mocopula_el([0.1, 0.2, 0.3], 2, [1., 2., 3.], s) ≈ [0.20082988502465082, 0.1344421423967149]
  end
  @testset "example on larger data" begin
    Random.seed!(43)
    x = simulate_copula(100000, MarshallOlkinCopula([1.1, 0.2, 0.6]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    a2 = 0.6/0.8
    a1 = 0.6/1.7
    @test corkendall(x)[1,2]≈ a1*a2/(a1+a2-a1*a2) atol=1.0e-3
    @test tail(x[:,1], x[:,2], "r") ≈ a1 atol=1.0e-1
    Random.seed!(42)
    x = simulate_copula(20000, MarshallOlkinCopula([1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  end
end

@testset "tests on Big Float" begin
  α = 0.025
  if false
    Random.seed!(1234)
    Σ = BigFloat.([1. 0.; 0. 1.])
    x = simulate_copula(1000, GaussianCopula(Σ))

    println(typeof(x) == Array{BigFloat,2})
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

    Random.seed!(1234)
    Σ = BigFloat.([1. 0.; 0. 1.])
    x = simulate_copula(1000, StudentCopula(Σ, 2))

    println(typeof(x) == Array{BigFloat,2})
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  end

  λs = BigFloat.([1., 2., 3.])
  Random.seed!(1234)
  x = simulate_copula(1000, MarshallOlkinCopula(λs))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

  a = b = BigFloat(0.4)
  x = simulate_copula(1000, FrechetCopula(2, a, b))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
end
