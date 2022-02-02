@testset "logseries dist" begin
  @test logseriescdf(0.01)[1:3] ≈ [0.0, 0.994992, 0.999967] atol=1.0e-5
  @test logseriescdf(0.000000000001)[1:5] ≈ [0., 1., 1., 1., 1.] atol=1.0e-2
  #@test logseriesquantile(0.9, [0.25, 0.5, 0.75]) == [1, 2, 5]
  Random.seed!(43)
  w = logseriescdf(0.4)
  v = pmap(i -> findlast(w .< i), rand(1000000))
  @test mean(v) ≈ 1.304 atol=1.0e-2
  @test std(v) ≈ 0.687 atol=1.0e-2
  @test skewness(v) ≈ 3.1 atol=1.0e-2
  @test kurtosis(v) ≈ 13.5 atol=1.0
end
@testset "stable levy dist" begin
  Random.seed!(43)
  rng = StableRNG(123)
  @test levyel(2., rand(rng), rand(rng)) ≈ 0.592404 atol=1.0e-5

  Random.seed!(43)
  rng = StableRNG(123)
  @test tiltedlevygen(0.2, 2.; rng = rng) ≈ 0.014182 atol=1.0e-5

  Random.seed!(43)
  rng = StableRNG(123)
  @test tiltedlevygen(0.6, 2.; rng = rng) ≈ 0.12764 atol=1.0e-5

end
@testset "nested copulas data generators" begin
  @test Ginv(0.5, 0.5) ≈ 1.2732395447351625
  @test InvlaJ(4, 0.5) ≈ 0.7265625
  @test sampleInvlaJ(0.5, 0.5) == 1
  @test sampleInvlaJ(0.5, 0.8) == 8
  Random.seed!(43)
  rng = StableRNG(123)
  w = logseriescdf(1-exp(-4.))
  @test elInvlaF(4., 2., w; rng = rng) == 1

  Random.seed!(43)
  rng = StableRNG(123)
  @test elInvlaF(4., .5, w; rng = rng) == 3
  @test nestedfrankgen(4., 3., 1, w; rng = rng) == 1

end
