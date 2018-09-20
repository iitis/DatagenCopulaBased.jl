α = 0.025

@testset "multiprocessing tests" begin
  addprocs(2)
  eval(Expr(:toplevel, :(@everywhere using DatagenCopulaBased)))
  eval(Expr(:toplevel, :(@everywhere import DatagenCopulaBased: nestedfrankgen)))
  eval(Expr(:toplevel, :(@everywhere using Random)))
  eval(Expr(:toplevel, :(@everywhere Random.seed!(44))))
  @test nestedfrankgen(5., 3., [1, 1, 2]) == [9, 54, 63]
  x = nestedarchcopulagen(250000, [3, 2],  [8., 10.], 2., "frank", 2)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
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
end
