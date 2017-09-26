
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

@testset "subcopulas" begin
  srand(43)
  cov = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.]
  x = gcopulagen(cov, 3)
  x1 = copy(x)
  g2tsubcopula!(x1, cov, [1,2])
  @test x1 ≈ [0.558652  0.719921  0.794493; 0.935573  0.922409  0.345177; 0.217512  0.174138  0.123049] atol=1.0e-5
  v = clcopappend(x[:,2], 0.5)
  @test v ≈ [0.31555, 0.846364, 0.0132052] atol=1.0e-5
end

@testset "helpers" begin
  v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
  v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
  @test lefttail(v1, v2, 0.1) ≈ 0.5
  @test righttail(v1, v2, 0.9) ≈ 0.5
end



@testset "probabilistic tests" begin
  cov = [1. 0.5; 0.5 1.]
  x = gcopulagen(cov, 100000)
  v = clcopappend(x[:,2], 0.8)
  y = copy(x)
  ν = 6
  g2tsubcopula!(y, cov, [1,2])
  xt = tcopulagen(cov, 100000, ν);
  @testset "tests for uniform distribution" begin
    srand(43)
    quant = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    @test quantile(x[:,1], quant) ≈ quant atol=1.0e-2
    @test quantile(x[:,2], quant) ≈ quant atol=1.0e-2
    @test quantile(v, quant) ≈ quant atol=1.0e-2
    @test quantile(y[:,1], quant) ≈ quant atol=1.0e-2
    @test quantile(y[:,2], quant) ≈ quant atol=1.0e-2
    @test quantile(xt[:,1], quant) ≈ quant atol=1.0e-2
    @test quantile(xt[:,2], quant) ≈ quant atol=1.0e-2
  end
  @testset "tail test" begin
    @test lefttail(x[:,1], x[:,2], 0.001) ≈ 0 atol=1.0e-1
    @test righttail(x[:,1], x[:,2], 0.999) ≈ 0 atol=1.0e-1
    d = TDist(ν+1)
    rho = cov[1,2]
    λ = 2*pdf(d, -sqrt.((ν+1)*(1-rho)/(1+rho)))
    @test lefttail(xt[:,1], xt[:,2], 0.001) ≈ λ atol=1.0e-1
    @test righttail(xt[:,1], xt[:,2], 0.999) ≈ λ atol=1.0e-1
  end
end
