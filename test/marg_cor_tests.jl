α = 0.025

@testset "transform marginals" begin
  x1 = [0.2 0.4; 0.4 0.6; 0.6 0.8]
  convertmarg!(x1, TDist, [[10],[10]])
  @test x1 ≈ [-0.879058  -0.260185; -0.260185 0.260185; 0.260185 0.879058] atol=1.0e-5
  Random.seed!(43)
  x = rand(10000, 2)
  convertmarg!(x, Normal, [[0., 2.],[0., 3.]])
  @test pvalue(ExactOneSampleKSTest(x[:,1],Normal(0,2))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2],Normal(0,3))) > α
  Random.seed!(43)
  @test_throws AssertionError convertmarg!(randn(1000, 2), Normal)
end

@testset "generate corelation matrix" begin
  Random.seed!(43)
  @test random_unit_vector(2) ≈ [0.2414027539276207, 0.9704250153392381]

  Random.seed!(43)
  @test cormatgen(2) ≈ [1.0 0.660768; 0.660768 1.0] atol=1.0e-5

  @test cormatgen_constant(3, 0.3) == [1 0.3 0.3; 0.3 1 0.3; 0.3 0.3 1]
  @test cormatgen_toeplitz(3, 0.3) == [1 0.3 0.09; 0.3 1 0.3; 0.09 0.3 1]
  @test cormatgen_two_constant(4, 0.5, 0.2) == [1. .5 .2 .2; .5 1. .2 .2; .2 .2 1. .2; .2 .2 .2 1.]

  @test cormatgen_constant_noised(10, 0.2, ϵ=0.) == cormatgen_constant(10, 0.2)
  @test all(eigvals(cormatgen_constant_noised(10, 0.2)) .> 0)
  @test diag(cormatgen_constant_noised(10, 0.2)) ≈ ones(10)

  @test cormatgen_toeplitz_noised(10, 0.2, ϵ=0.) == cormatgen_toeplitz(10, 0.2)
  @test all(eigvals(cormatgen_toeplitz_noised(10, 0.2)) .> 0)
  @test diag(cormatgen_toeplitz_noised(10, 0.2)) ≈ ones(10)

  @test cormatgen_two_constant_noised(10, 0.8, 0.2, ϵ=0.) == cormatgen_two_constant(10, 0.8, 0.2)
  @test all(eigvals(cormatgen_two_constant_noised(10, 0.8, 0.2)) .> 0)
  @test diag(cormatgen_two_constant_noised(10, 0.8, 0.2)) ≈ ones(10)

  Random.seed!(43)
  @test cormatgen_rand(2) ≈ [1.0 0.879086; 0.879086 1.0] atol=1.0e-5
  @test all(eigvals(cormatgen_rand(10)) .> 0)
  @test diag(cormatgen_rand(10)) ≈ ones(10)

end
