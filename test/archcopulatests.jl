α = 0.025

# Hepers

@testset "correlations" begin
  @testset "kendall's cor" begin
    @test Debye(0.5, 1) ≈ 0.8819271567906056
    @test τ2λ([0.4], [4.2, 1.2]) ≈ [4.2, 1.2, 3.6]
    @test τ2θ(0.6, "frank") ≈ 7.929642284264058
    @test frankτ2θ(0.6) ≈ 7.929642284264058
    @test τ2θ(0.5, "gumbel") ≈ 2.
    @test τ2θ(1/3, "clayton") ≈ 1.
    @test τ2θ(1/4, "amh") ≈ 0.8384520912688538
    @test AMHτ2θ(1/4) ≈ 0.8384520912688538
  end
  @testset "pearson cor" begin
    @test dilog(0.5) ≈ 0.5822405264650125
    @test ρ2θ(1/3, "clayton") ≈ 0.58754 atol=1.0e-5
    @test ρ2θ(1/3, "frank") ≈ 2.1164969117225363
    @test ρ2θ(0.5, "gumbel") ≈ 1.5410704204332681
    @test ρ2θ(0.2, "amh") ≈ 0.5168580913147318
    @test frankθ(1/3) ≈ 2.1164969117225363
    @test AMHθ(0.2) ≈ 0.5168580913147318
  end
end
@testset "archimedean copulas axiliary functions" begin
  @test getV0(2., [0.2, 0.4, 0.6, 0.8], "clayton") ≈ [0.0641848, 0.274996, 0.708326, 1.64237] atol=1.0e-4
  @test phi([0.2 0.6; 0.4 0.8], 2., "clayton") ≈ [0.845154  0.6742; 0.745356  0.620174] atol=1.0e-4
  c = copulagen("clayton", [0.2 0.4 0.8; 0.2 0.8 0.6; 0.3 0.9 0.6], 1.)
  @test c ≈ [0.5 0.637217; 0.362783 0.804163; 0.432159 0.896872] atol=1.0e-5
  @test useτ(0.5, "clayton") == 2.
  @test useρ(0.75, "gumbel") ≈ 2.285220798876495
end

@testset "archimedean copulas exceptions" begin
  @test_throws(AssertionError, testθ(0.5, "gumbel"))
  @test_throws(AssertionError, useρ(0.6, "amh"))
  @test_throws(AssertionError, useτ(0.45, "amh"))
  @test_throws(AssertionError, archcopulagen(100000, 4, -0.6, "frank"))
end

@testset "gumbel copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 2., "gumbel");
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
  @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.
  @test corkendall(x) ≈ [1. 1/2 1/2; 1/2 1. 1/2; 1/2 1/2 1.] atol=1.0e-2
  srand(43)
  x = archcopulagen(500000, 2, 1.5, "gumbel"; rev = true)
  @test tail(x[:,1], x[:,2], "l") ≈ 2-2^(1/1.5) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0.
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  srand(43)
  x = archcopulagen(500000, 2, 0.5, "gumbel"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-4
  srand(43)
  x = archcopulagen(500000, 2, 0.5, "gumbel"; cor = "pearson")
  @test cor(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
end
@testset "clayton copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 1., "clayton");
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
  @test tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x) ≈ [1. 1/3 1/3; 1/3 1. 1/3; 1/3 1/3 1.] atol=1.0e-2
  srand(43)
  x = archcopulagen(500000, 2, 0.5, "clayton"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-3
  srand(43)
  x = archcopulagen(500000, 2, -0.9, "clayton")
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-3
end
@testset "frank copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 0.8, "frank")
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0
  srand(43)
  x = archcopulagen(500000, 2, 0.2, "frank"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.2; 0.2 1.] atol=1.0e-3
end
@testset "Ali-Mikhail-Haq copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 0.8, "amh")
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x)[1:2, 1:2] ≈ [1. 0.23373; 0.23373 1.] atol=1.0e-3
  x = archcopulagen(500000, 2, 0.25, "amh"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.25; 0.25 1.] atol=1.0e-3
end
