α = 0.025

@testset "gaussian copula" begin
  @testset "exceptions" begin
    @test_throws DomainError Gaussian_cop([1. 0.; .1 1.])
    @test_throws DomainError Gaussian_cop([1. 2.; .2 1.])
    @test_throws DomainError Gaussian_cop([2. 1.; 1. 1.])
  end
  @testset "examples on data" begin
    Random.seed!(43)
    if VERSION <= v"1.7"
      @test simulate_copula(1, Gaussian_cop([1. 0.; 0. 1.])) ≈ [0.589188  0.817617] atol=1.0e-5
    else
      @test simulate_copula(1, Gaussian_cop([1. 0.; 0. 1.])) ≈ [0.623817 0.0518950] atol=1.0e-5
    end
    Random.seed!(43)
    x = simulate_copula(350000, Gaussian_cop([1. 0.5; 0.5 1.]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0
  end

end
@testset "t-student copula" begin
  @testset "exceptions" begin
    @test_throws DomainError Student_cop([1. 0.; .1 1.], 2)
    @test_throws DomainError Student_cop([1. 2.; .2 1.], 2)
    @test_throws DomainError Student_cop([2. 1.; 1. 1.], 2)
    @test_throws DomainError Student_cop([1. 0.; 0. 1.], 0)
  end

  @testset "example on data" begin
    Random.seed!(43)
    @test simulate_copula(1, Student_cop([1. 0.; 0. 1.], 1)) ≈ [0.936433  0.983987] atol=1.0e-5
    ν = 10
    rho = 0.5
    λ = 2*pdf(TDist(ν+1), -sqrt.((ν+1)*(1-rho)/(1+rho)))
    Random.seed!(43)
    xt = simulate_copula(350000, Student_cop([1. rho; rho 1.], ν))
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
    @test_throws DomainError Frechet_cop(3, -0.8)
    @test_throws DomainError Frechet_cop(1, 0.8)

    @test_throws AssertionError Frechet_cop(3, 0.8, 0.1)
    @test_throws DomainError Frechet_cop(2, 0.8, 0.5)
    @test_throws DomainError Frechet_cop(2, -0.2, 0.8)
    @test_throws DomainError Frechet_cop(2, 0.8, -0.5)

    u = zeros(3, 5)
    cop = Frechet_cop(3, 0.8)
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
    @test simulate_copula(1, Frechet_cop(4, 1.)) ≈ [0.9248760  0.9248760  0.9248760  0.9248760] atol=1.0e-5
    Random.seed!(43)
    @test frechet(1., rand(1,4); rng = Random.GLOBAL_RNG) ≈ [0.9248760  0.9248760  0.9248760  0.9248760] atol=1.0e-5
    Random.seed!(43)
    @test simulate_copula(1, Frechet_cop(2, 0.4, 0.4)) ≈ [0.1809752 0.7753771] atol=1.0e-5

    u = zeros(1,4)
    u1 = zeros(1,2)
    Random.seed!(43)
    simulate_copula!(u, Frechet_cop(4, 1.))
    @test u ≈ [0.9248760  0.9248760  0.9248760  0.9248760] atol=1.0e-5
    Random.seed!(43)
    simulate_copula!(u1, Frechet_cop(2, 0.4, 0.4))
    @test u1 ≈ [0.1809752 0.7753771] atol=1.0e-5
  end
  @testset "examples on larger data" begin
    Random.seed!(43)
    x = simulate_copula(350000, Frechet_cop(3, 0.3))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
    @test tail(x[:,1], x[:,2], "l") ≈ 0.3 atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r") ≈ 0.3 atol=1.0e-1
    @test corspearman(x) ≈ [1. 0.3 0.3; 0.3 1. 0.3; 0.3 0.3 1.] atol=1.0e-2


    x = simulate_copula(350000, Frechet_cop(3, 1.))
    ret = (x[:, 1] .<  0.2).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
    @test length(findall(ret))/size(x,1) ≈ 0.2 atol=1.0e-3
    ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
    @test length(findall(ret))/size(x,1) ≈ 0.3 atol=1.0e-3
    ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.9).*(x[:, 3] .<  0.5)
    @test length(findall(ret))/size(x,1) ≈ 0.5 atol=1.0e-2
    Random.seed!(43)
    x = simulate_copula(1500000, Frechet_cop(2, .0))
    @test corspearman(x) ≈ [1. 0.; 0. 1.] atol=1.0e-2

    x = simulate_copula(350000, Frechet_cop(2, 0.8, 0.1));
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test corspearman(x) ≈ [1. 0.7; 0.7 1.] atol=1.0e-2
    @test tail(x[:,1], x[:,2], "l") ≈ 0.8 atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r") ≈ 0.8 atol=1.0e-1
  end
end


@testset "Marshall-Olkin copula" begin
  @testset "helpers and exceptions" begin
    @test_throws AssertionError Marshall_Olkin_cop([1., -2., 3.])
    @test_throws AssertionError Marshall_Olkin_cop([1., 3.])
    @test τ2λ([0.4], [4.2, 1.2]) ≈ [4.2, 1.2, 3.6]
    @test τ2λ([0.5, 0.6, 0.5], [1., 2., 3., 4.]) ≈ [1, 2, 3, 0, 0.5, 0, 4]
    @test moρ2τ(0.6) ≈ 0.5 atol=1.0e-2

    u = zeros(3,3)
    cop = Marshall_Olkin_cop([1.,2.,3.])
    @test_throws AssertionError simulate_copula!(u, cop)
  end
  @testset "small example" begin
    Random.seed!(43)
    @test simulate_copula(1, Marshall_Olkin_cop([1., 2., 3.])) ≈ [0.854724  0.821831] atol=1.0e-5

    u = zeros(1,2)
    Random.seed!(43)
    simulate_copula!(u, Marshall_Olkin_cop([1., 2., 3.]))
    @test u ≈ [0.854724  0.821831] atol=1.0e-5

    m = [0.252982 0.201189;  0.464758 0.409039; 0.585662 0.5357]
    @test mocopula([0.2 0.3 0.4; 0.3 0.4 0.6; 0.4 0.5 0.7], 2, [1., 1.5, 2.]) ≈ m atol=1.0e-4

    s = collect(combinations(1:2))
    mocopula_el([0.1, 0.2, 0.3], 2, [1., 2., 3.], s) ≈ [0.20082988502465082, 0.1344421423967149]
  end
  @testset "example on larger data" begin
    Random.seed!(43)
    x = simulate_copula(100000, Marshall_Olkin_cop([1.1, 0.2, 0.6]))
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    a2 = 0.6/0.8
    a1 = 0.6/1.7
    @test corkendall(x)[1,2]≈ a1*a2/(a1+a2-a1*a2) atol=1.0e-3
    @test tail(x[:,1], x[:,2], "r") ≈ a1 atol=1.0e-1
    Random.seed!(42)
    x = simulate_copula(100000, Marshall_Olkin_cop([1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1]))
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
    x = simulate_copula(1000, Gaussian_cop(Σ))

    println(typeof(x) == Array{BigFloat,2})
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

    Random.seed!(1234)
    Σ = BigFloat.([1. 0.; 0. 1.])
    x = simulate_copula(1000, Student_cop(Σ, 2))

    println(typeof(x) == Array{BigFloat,2})
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  end

  λs = BigFloat.([1., 2., 3.])
  Random.seed!(1234)
  x = simulate_copula(1000, Marshall_Olkin_cop(λs))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α

  a = b = BigFloat(0.4)
  x = simulate_copula(1000, Frechet_cop(2, a, b))
  @test typeof(x) == Array{BigFloat,2}
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
end
