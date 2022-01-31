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
  if VERSION <= v"1.7"
    @test levyel(2., rand(), rand()) ≈ 0.181703 atol=1.0e-5
    Random.seed!(43)
    @test tiltedlevygen(0.2, 2.; rng = Random.GLOBAL_RNG) ≈ 0.00409133 atol=1.0e-5
    Random.seed!(43)
    @test tiltedlevygen(0.6, 2.; rng = Random.GLOBAL_RNG) ≈ 0.036822009 atol=1.0e-5
  else
    @test levyel(2., rand(), rand()) ≈ 0.059716 atol=1.0e-5
    Random.seed!(43)
    @test tiltedlevygen(0.2, 2.; rng = Random.GLOBAL_RNG) ≈ 0.032748 atol=1.0e-5
    Random.seed!(43)
    @test tiltedlevygen(0.6, 2.; rng = Random.GLOBAL_RNG) ≈ 0.294735 atol=1.0e-5
  end

end
@testset "nested copulas data generators" begin
  @test Ginv(0.5, 0.5) ≈ 1.2732395447351625
  @test InvlaJ(4, 0.5) ≈ 0.7265625
  @test sampleInvlaJ(0.5, 0.5) == 1
  @test sampleInvlaJ(0.5, 0.8) == 8
  Random.seed!(43)
  w = logseriescdf(1-exp(-4.))
  @test elInvlaF(4., 2., w; rng = Random.GLOBAL_RNG) == 7
  Random.seed!(43)
  @test elInvlaF(4., .5, w; rng = Random.GLOBAL_RNG) == 16
  @test nestedfrankgen(4., 3., 1, w; rng = Random.GLOBAL_RNG) == 6
end
