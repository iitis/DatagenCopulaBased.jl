α = 0.025

@testset "archimedean bivariate copula helpers" begin
  srand(43)
  @test pvalue(ExactOneSampleKSTest(rand2cop(rand(500000), 0.5, "clayton"), Uniform(0,1))) > α
  srand(43)
  @test pvalue(ExactOneSampleKSTest(rand2cop(rand(500000), 0.5, "frank"), Uniform(0,1))) > α
  srand(43)
  @test pvalue(ExactOneSampleKSTest(rand2cop(rand(500000), 0.5, "amh"), Uniform(0,1))) > α
  srand(43)
  @test rand2cop([0.815308, 0.894269], 0.5, "clayton") ≈ [0.292041, 0.836167] atol=1.0e-5
end

@testset "bivariate archimedean copulas exceptions" begin
  @test_throws(AssertionError, testbivθ(-2., "clayton"))
  @test_throws(AssertionError, usebivρ(-.9, "amh", "pearson"))
  @test_throws(AssertionError, bivariatecopgen(100000, [1.1, 1.6], "gumbel"))
end

@testset "clayton bivariate copulas" begin
  srand(43)
  x = bivariatecopgen(500000, [-0.9, 3., 2], "clayton")
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test tail(x[:,3], x[:,4], "l") ≈ 1/(2^(1/2)) atol=1.0e-1
  @test tail(x[:,3], x[:,4], "r", 0.0001) ≈ 0
  @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-3
  @test corkendall(x)[2,3] ≈ 3/(2+3) atol=1.0e-3
  srand(43)
  x = bivariatecopgen(500000, [0.6, -0.2], "clayton"; cor = "pearson")
  @test cor(x[:,1], x[:,2]) ≈ 0.6 atol=1.0e-2
  @test cor(x[:,2], x[:,3]) ≈ -0.2 atol=1.0e-2
  srand(43)
  x = bivariatecopgen(500000, [0.6, -0.2], "clayton"; cor = "kendall")
  @test corkendall(x[:,1], x[:,2]) ≈ 0.6 atol=1.0e-3
  @test corkendall(x[:,2], x[:,3]) ≈ -0.2 atol=1.0e-3
  srand(43)
  x = bivariatecopgen(500000, [-0.9, 3., 2.], "clayton"; rev = true)
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test tail(x[:,3], x[:,4], "r") ≈ 1/(2^(1/2)) atol=1.0e-1
  @test tail(x[:,3], x[:,4], "l", 0.0001) ≈ 0
end

@testset "frank bivariate copulas" begin
  srand(43)
  x = bivariatecopgen(500000, [4., 11., 0.5, -12.], "frank")
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,4], x[:,3], "r", 0.0001) ≈ 0
  srand(43)
  x = bivariatecopgen(500000, [0.8, 0.3, -0.5], "frank"; cor = "pearson")
  @test cor(x[:,1], x[:,2]) ≈ 0.8 atol=1.0e-3
  @test cor(x[:,2], x[:,3]) ≈ 0.3 atol=1.0e-3
end
@testset "Ali-Mikhail-Haq bivariate subcopulas" begin
  srand(43)
  x = bivariatecopgen(500000, [0.3, 0.6, 1.], "amh")
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,3], x[:,4], "l") ≈ 0.5 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x)[3,4] ≈ 1/3 atol=1.0e-5
  @test corkendall(x)[1,2] ≈ 0.072 atol=1.0e-3
  srand(43)
  x = bivariatecopgen(500000, [0.45, 0.3], "amh"; cor = "pearson")
  @test cor(x[:,1], x[:,2]) ≈ 0.45 atol=1.0e-2
  @test cor(x[:,2], x[:,3]) ≈ 0.3 atol=1.0e-2
end

@testset "bivariate frechet copula" begin
  @test fncopulagen([0.2, 0.4], [0.1, 0.1], [0.2 0.4 0.6; 0.3 0.5 0.7]) == [0.6 0.4 0.2; 0.7 0.5 0.3]
  srand(43)
  x = bivfrechetcopulagen(500000, [0.9, 0.6, 0.2])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test cor(x) ≈ [1. 0.9 0.6 0.2; 0.9 1. 0.6 0.2; 0.6 0.6 1. 0.2; 0.2 0.2 0.2 1.] atol=1.0e-2
  @test tail(x[:,1], x[:,2], "r") ≈ 0.9 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l") ≈ 0.9 atol=1.0e-1
  @test tail(x[:,1], x[:,4], "r") ≈ 0.2 atol=1.0e-1
  srand(43)
  x = bivfrechetcopulagen(500000, [0.8, 0.5], [0.2, 0.3]);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  cor(x) ≈ [1. 0.6 0.2; 0.6 1. 0.2; 0.2 0.2 1.]
  @test tail(x[:,1], x[:,2], "r") ≈ 0.8 atol=1.0e-1
  @test tail(x[:,2], x[:,3], "r") ≈ 0.5 atol=1.0e-1
end

@testset "bivariate copula mixture" begin
  srand(43)
  Σ = cormatgen(15, 0.8, true,true)
  d=["clayton" => [2,3,4,5,6], "amh" => [1,14], "frank" => [7,8]]
  x = bivariatecopulamix(100000, Σ, d);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
end
