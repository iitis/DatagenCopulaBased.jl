
@testset "generate data" begin
  @testset "axiliary functions" begin
    @test invers_gen([1., 2., 3., 4., 5.], 3.2) ≈ [0.638608, 0.535014, 0.478181, 0.44034, 0.412558] atol=1.0e-5
    srand(43)
    cor, cov = covmatgen(3)
    @test cor ≈ [0.274784 -0.068671 0.933668; -0.068671 0.741411 0.657473; 0.933668 0.657473 7.93625] atol=1.0e-5
    @test cov ≈ [1.0 -0.152142 0.63225; -0.152142 1.0 0.271045; 0.63225 0.271045 1.0] atol=1.0e-5
  end
  @testset "generate data from copuls" begin
    srand(43)
    @test clcopulagen(2,2) ≈ [0.991747 0.20762; 2.96931 2.79434] atol=1.0e-5
    srand(43)
    @test tcopulagen([[1. 0.5];[0.5 1.]], 2) ≈ [0.581625 0.792144; 0.76935 0.968669] atol=1.0e-5
    srand(43)
    @test gcopulagen([[1. 0.5];[0.5 1.]], 2) ≈ [0.589188 0.815308; 0.708285 0.924962] atol=1.0e-5
  end
  @testset "transform marginals" begin
    @test u2normal([0.2 0.4; 0.4 0.6; 0.6 0.8]) ≈ [-0.841621 -0.253347; -0.253347 0.253347; 0.253347 0.841621] atol=1.0e-5
    @test u2tdist([0.2 0.4; 0.4 0.6; 0.6 0.8], 10) ≈ [-0.879058  -0.260185; -0.260185 0.260185; 0.260185 0.879058] atol=1.0e-5
  end
  @testset "generates data given copula nad marginal" begin
    srand(43)
    @test gcopulatmarg([1. 0.5;0.5 1.], 2, 10) ≈ [0.231461 0.939967; 0.566673 1.55891] atol=1.0e-5
    srand(43)
    @test tcopulagmarg([1. 0.5;0.5 1.], 2, 10) ≈ [0.200129 0.784859; 0.87 2.0739] atol=1.0e-5
    srand(43)
    @test tdistdat([1. 0.5;0.5 1.], 2, 10) ≈ [0.205402 0.81778; 0.909864 2.388] atol=1.0e-4
    srand(43)
    @test normdist([1. 0.5;0.5 1.], 2, 10) ≈ [0.225457 0.897627; 0.548381 1.43926] atol=1.0e-5
  end
end
