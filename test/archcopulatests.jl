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
@testset "archimedean copulas axiliary functions" begin
  @test getV0(2., [0.2, 0.4, 0.6, 0.8], "clayton") ≈ [0.0320924, 0.137498, 0.354163, 0.821187] atol=1.0e-4
  @test_throws AssertionError getV0(2., [0.2, 0.4, 0.6, 0.8], "clayto")
  @test phi([0.2 0.6; 0.4 0.8], 2., "clayton") ≈ [0.912871  0.790569; 0.845154  0.745356] atol=1.0e-4
  @test_throws AssertionError phi([0.2 0.6; 0.4 0.8], 2., "clayto")
  c = arch_gen("clayton", [0.2 0.4 0.8; 0.2 0.8 0.6; 0.3 0.9 0.6], 1.)
  @test c ≈ [0.5 0.637217; 0.362783 0.804163; 0.432159 0.896872] atol=1.0e-5
  @test useτ(0.5, "clayton") == 2.
  @test useρ(0.75, "gumbel") ≈ 2.285220798876495
  @test getθ4arch(0.5, "gumbel", "Spearman") ≈ 1.541070420842913
  @test getθ4arch(0.5, "gumbel", "Kendall") ≈ 2.0
  @test getθ4arch(1.5, "gumbel", "") == 1.5
  @test_throws AssertionError getθ4arch(1.5, "gumbel", "Pearson")
end

@testset "archimedean copulas exceptions" begin
  @test_throws DomainError testθ(0.5, "gumbel")
  @test_throws DomainError useρ(0.6, "amh")
  @test_throws DomainError useτ(0.45, "amh")
  @test_throws DomainError archcopulagen(100000, 4, -0.6, "frank")
end

@testset "gumbel copula" begin
  Random.seed!(43)
  x = simulate_copula(500000, gumbel, 3, 2.)
  Random.seed!(43)
  x1 = gumbel(500000, 3, 2.)
  Random.seed!(43)
  x2 = archcopulagen(500000, 3, 2., "gumbel")
  @test norm(x - x1) ≈ 0
  @test norm(x - x2) ≈ 0
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
  @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.
  @test corkendall(x) ≈ [1. 1/2 1/2; 1/2 1. 1/2; 1/2 1/2 1.] atol=1.0e-2
  Random.seed!(43)
  x = simulate_copula(500000, rev_gumbel, 2, 1.5)
  Random.seed!(43)
  x1 = archcopulagen(500000, 2, 1.5, "gumbel"; rev = true)
  @test norm(x - x1) ≈ 0
  @test tail(x[:,1], x[:,2], "l") ≈ 2-2^(1/1.5) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0.
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  Random.seed!(43)
  x = simulate_copula(500000, gumbel, 2, 0.5; cor = "Kendall")
  @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-4
  Random.seed!(43)
  x = simulate_copula(500000, gumbel, 2, 0.5; cor = "Spearman")
  @test corspearman(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
end
@testset "clayton copula" begin
  Random.seed!(43)
  x2 = clayton(500000, 3, 1.)
  Random.seed!(43)
  x1 = archcopulagen(500000, 3, 1., "clayton")
  Random.seed!(43)
  x = simulate_copula(500000, clayton, 3, 1.)
  @test norm(x - x1) ≈ 0
  @test norm(x - x2) ≈ 0
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
  @test tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x) ≈ [1. 1/3 1/3; 1/3 1. 1/3; 1/3 1/3 1.] atol=1.0e-2
  Random.seed!(43)
  x = simulate_copula(500000, clayton, 2, 0.5; cor = "Kendall")
  @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-3
  Random.seed!(43)
  x = simulate_copula(500000, clayton, 2, -0.9)
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-3

  Random.seed!(43)
  x = simulate_copula(500000, rev_clayton, 2, -0.9)
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-3
  Random.seed!(43)
  x1 = archcopulagen(500000, 2, -0.9, "clayton"; rev = true)
  @test norm(x - x1) ≈ 0
end
@testset "frank copula" begin
  Random.seed!(43)
  x = simulate_copula(500000, frank, 3, 0.8)
  Random.seed!(43)
  x1 = frank(500000, 3, 0.8)
  Random.seed!(43)
  x2 = archcopulagen(500000, 3, 0.8, "frank")
  @test norm(x - x1) ≈ 0
  @test norm(x - x2) ≈ 0
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0
  Random.seed!(43)
  x = simulate_copula(500000, frank, 2, 0.2; cor = "Kendall")
  @test corkendall(x) ≈ [1. 0.2; 0.2 1.] atol=1.0e-3
  Random.seed!(43)
  x = simulate_copula(500000, frank, 2, -2.)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
end
@testset "Ali-Mikhail-Haq copula" begin
  Random.seed!(43)
  x = simulate_copula(500000, amh, 3, 0.8)
  Random.seed!(43)
  x1 = amh(500000, 3, 0.8)
  Random.seed!(43)
  x2 = archcopulagen(500000, 3, 0.8, "amh")
  @test norm(x - x1) ≈ 0
  @test norm(x - x2) ≈ 0
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x)[1:2, 1:2] ≈ [1. 0.23373; 0.23373 1.] atol=1.0e-3
  x = simulate_copula(500000, amh, 2, 0.25; cor = "Kendall")
  @test corkendall(x) ≈ [1. 0.25; 0.25 1.] atol=1.0e-3

  Random.seed!(43)
  x = simulate_copula(500000, rev_amh, 3, 0.8)
  Random.seed!(43)
  x2 = archcopulagen(500000, 3, 0.8, "amh"; rev = true)
  @test norm(x - x2) ≈ 0
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

  Random.seed!(43)
  x = simulate_copula(500000, rev_amh, 2, -0.4)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
end
