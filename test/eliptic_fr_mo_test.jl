α = 0.025

@testset "gaussian copula" begin
  Random.seed!(43)
  @test gaussian_cop(1, [1. 0.; 0. 1.]) ≈ [0.589188  0.817617] atol=1.0e-5
  Random.seed!(43)
  x = simulate_copula1(500000, gaussian_cop1([1. 0.5; 0.5 1.]))
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0

  # compare old and new dispatching
  Random.seed!(43)
  x1 = gausscopulagen(500000, [1. 0.5; 0.5 1.])
  @test norm(x-x1) == 0.
  Random.seed!(43)
  x2 = gaussian_cop(500000, [1. 0.5; 0.5 1.])
  @test norm(x-x2) == 0.
end
@testset "t-student copula" begin
  Random.seed!(43)
  @test tstudent_cop(1, [1. 0.; 0. 1.], 1) ≈ [0.936433  0.983987] atol=1.0e-5
  ν = 10
  rho = 0.5
  λ = 2*pdf(TDist(ν+1), -sqrt.((ν+1)*(1-rho)/(1+rho)))
  Random.seed!(43)
  xt = simulate_copula1(500000, tstudent_cop1([1. 0.5; 0.5 1.], ν))
  #xt = simulate_copula(500000, tstudent_cop, [1. rho; rho 1.], ν);
  @test pvalue(ExactOneSampleKSTest(xt[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(xt[:,2], Uniform(0,1))) > α
  @test tail(xt[:,1], xt[:,2], "l") ≈ λ atol=1.0e-1
  @test tail(xt[:,1], xt[:,2], "r") ≈ λ atol=1.0e-1
  # compare old and new dispatching
  Random.seed!(43)
  xt1 = tstudentcopulagen(500000, [1. rho; rho 1.], ν);
  @test norm(xt-xt1) == 0.
  Random.seed!(43)
  xt2 = tstudent_cop(500000, [1. rho; rho 1.], ν);
  @test norm(xt-xt2) == 0.

  convertmarg!(xt, Normal)
  @test cor(xt) ≈ [1. rho; rho 1.] atol=1.0e-2

  Random.seed!(43)
  xtt = simulate_copula(500000, tstudent_cop, [4. 2*rho; 2*rho 1.], ν);
  @test pvalue(ExactOneSampleKSTest(xtt[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(xtt[:,2], Uniform(0,1))) > α
  @test tail(xtt[:,1], xtt[:,2], "l") ≈ λ atol=1.0e-1
  @test tail(xtt[:,1], xtt[:,2], "r") ≈ λ atol=1.0e-1
  convertmarg!(xtt, Normal)
  @test cov(xtt) ≈ [1. rho; rho 1.] atol=1.0e-2
end
@testset "Frechet copula" begin
  Random.seed!(43)
  @test frechet(1, 4, 1) ≈ [0.180975  0.180975  0.180975  0.180975] atol=1.0e-5
  Random.seed!(43)
  x = simulate_copula(500000, frechet, 3, 0.3)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l") ≈ 0.3 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r") ≈ 0.3 atol=1.0e-1
  @test corspearman(x) ≈ [1. 0.3 0.3; 0.3 1. 0.3; 0.3 0.3 1.] atol=1.0e-2

  # compare old and new dispatching
  Random.seed!(43)
  x1 = frechetcopulagen(500000, 3, 0.3)
  @test norm(x-x1) == 0.

  Random.seed!(43)
  x2 = frechet(500000, 3, 0.3)
  @test norm(x-x2) == 0.

  x = simulate_copula(500000, frechet, 3, 1.)
  ret = (x[:, 1] .<  0.2).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
  @test length(findall(ret))/size(x,1) ≈ 0.2 atol=1.0e-3
  ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
  @test length(findall(ret))/size(x,1) ≈ 0.3 atol=1.0e-3
  ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.9).*(x[:, 3] .<  0.5)
  @test length(findall(ret))/size(x,1) ≈ 0.5 atol=1.0e-3
  Random.seed!(43)
  x = simulate_copula(500000, frechet, 2, .0)
  @test corspearman(x) ≈ [1. 0.; 0. 1.] atol=1.0e-2
  x = simulate_copula(500000, frechet, 2, 0.8, 0.1);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test corspearman(x) ≈ [1. 0.7; 0.7 1.] atol=1.0e-2
  @test tail(x[:,1], x[:,2], "l") ≈ 0.8 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r") ≈ 0.8 atol=1.0e-1
end

@testset "Frechet copula exceptions" begin
  @test_throws AssertionError simulate_copula(1000, frechet, 3, 0.8, 0.1)
  @test_throws DomainError simulate_copula(1000, frechet, 2, 0.8, 0.5)
  @test_throws DomainError simulate_copula(1000, frechet, 3, -0.8)
end

@testset "Marshall-Olkin helpers" begin
  @test τ2λ([0.4], [4.2, 1.2]) ≈ [4.2, 1.2, 3.6]
  @test τ2λ([0.5, 0.6, 0.5], [1., 2., 3., 4.]) ≈ [1, 2, 3, 0, 0.5, 0, 4]
  @test moρ2τ(0.6) ≈ 0.5 atol=1.0e-2
end
@testset "Marshall-Olkin copula" begin
  Random.seed!(43)
  @test marshallolkin(1, [1., 2., 3.]) ≈ [0.854724  0.821831] atol=1.0e-5

  m = [0.252982 0.201189;  0.464758 0.409039; 0.585662 0.5357]
  @test mocopula([0.2 0.3 0.4; 0.3 0.4 0.6; 0.4 0.5 0.7], 2, [1., 1.5, 2.]) ≈ m atol=1.0e-4
  Random.seed!(43)
  x = simulate_copula(100000, marshallolkin, [1.1, 0.2, 0.6])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  a2 = 0.6/0.8
  a1 = 0.6/1.7
  @test corkendall(x)[1,2]≈ a1*a2/(a1+a2-a1*a2) atol=1.0e-3
  @test tail(x[:,1], x[:,2], "r") ≈ a1 atol=1.0e-1
  Random.seed!(42)
  x = simulate_copula(100000, marshallolkin, [1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α

  # compare old and new dispatching

  Random.seed!(42)
  x1 = marshallolkincopulagen(100000, [1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1])
  @test norm(x-x1) == 0.

  Random.seed!(42)
  x2 = marshallolkin(100000, [1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1])
  @test norm(x-x2) == 0.
end
