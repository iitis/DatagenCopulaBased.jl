α = 0.05

@testset "generate data" begin
  @testset "axiliary functions" begin
    @test invers_gen([1., 2., 3., 4., 5.], 3.2) ≈ [0.638608, 0.535014, 0.478181, 0.44034, 0.412558] atol=1.0e-5
    srand(43)
    @test cormatgen(2) ≈ [1.0 -0.901386; -0.901386 1.0] atol=1.0e-5
  end
  @testset "tail dependencies" begin
    v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
    v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
    @test lefttail(v1, v2, 0.1) ≈ 0.5
    @test righttail(v1, v2, 0.9) ≈ 0.5
  end
  @testset "transform marginals" begin
    x = [0.2 0.4; 0.4 0.6; 0.6 0.8]
    x1 = [0.2 0.4; 0.4 0.6; 0.6 0.8]
    convertmarg!(x, Normal)
    convertmarg!(x1, TDist, [[10],[10]])
    @test x ≈ [-0.841621 -0.253347; -0.253347 0.253347; 0.253347 0.841621] atol=1.0e-5
    @test x1 ≈ [-0.879058  -0.260185; -0.260185 0.260185; 0.260185 0.879058] atol=1.0e-5
    srand(43)
    x = rand(10000, 2)
    srand(43)
    x1 = rand(10000, 2)
    convertmarg!(x, Normal, [[0., 2.],[0., 3.]])
    convertmarg!(x1, TDist, [[10],[6]])
    @test pvalue(ExactOneSampleKSTest(x[:,1],Normal(0,2))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2],Normal(0,3))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,1],TDist(10))) > α
    @test pvalue(ExactOneSampleKSTest(x1[:,2],TDist(6))) > α
    srand(43)
    @test_throws AssertionError convertmarg!(randn(1000, 2), Normal)
  end
end

@testset "copulas" begin
  @testset "generate data from copuls" begin
    srand(43)
    @test claytoncopulagen(2,2) ≈ [0.629041  0.182246; 0.950303  0.942292] atol=1.0e-5
    srand(43)
    @test tstudentcopulagen(2, [[1. 0.5];[0.5 1.]]) ≈ [0.581625 0.792144; 0.76935 0.968669] atol=1.0e-5
    srand(43)
    @test gausscopulagen(2, [[1. 0.5];[0.5 1.]]) ≈ [0.589188 0.815308; 0.708285 0.924962] atol=1.0e-5
  end
  @testset "gaussian copula" begin
  srand(43)
    x = gausscopulagen(500000, [1. 0.5; 0.5 1.])
    @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
    @test copuladeftest(x[:,1], x[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
    @test lefttail(x[:,1], x[:,2], 0.001) ≈ 0 atol=1.0e-1
    @test righttail(x[:,1], x[:,2], 0.999) ≈ 0 atol=1.0e-1
  end
  @testset "t-student copula" begin
    ν = 10
    dt = TDist(ν+1)
    rho = 0.5
    λ = 2*pdf(dt, -sqrt.((ν+1)*(1-rho)/(1+rho)))
    srand(43)
    xt = tstudentcopulagen(500000, [1. 0.5; 0.5 1.], ν);
    @test pvalue(ExactOneSampleKSTest(xt[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xt[:,2], Uniform(0,1))) > α
    @test copuladeftest(xt[:,1], xt[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
    @test lefttail(xt[:,1], xt[:,2], 0.001) ≈ λ atol=1.0e-1
    @test righttail(xt[:,1], xt[:,2], 0.999) ≈ λ atol=1.0e-1
  end
  srand(43)
  xc = claytoncopulagen(500000, 2);
  @testset "clayton copula" begin
    @test pvalue(ExactOneSampleKSTest(xc[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xc[:,2], Uniform(0,1))) > α
    @test lefttail(xc[:,1], xc[:,2], 0.001) ≈ 0.5 atol=1.0e-1
    @test righttail(xc[:,1], xc[:,2], 0.999) ≈ 0 atol=1.0e-1
    @test copuladeftest(xc[:,1], xc[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
  end
end

@testset "subcopulas" begin
  @testset "generate data from subcopuls" begin
    srand(43)
    x = gausscopulagen(3, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.])
    x1 = copy(x)
    g2tsubcopula!(x1, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.], [1,2])
    @test x1 ≈ [0.558652  0.719921  0.794493; 0.935573  0.922409  0.345177; 0.217512  0.174138  0.123049] atol=1.0e-5
    v = g2clsubcopula(x[:,2], 0.5)
    @test v ≈ [0.31555, 0.846364, 0.0132052] atol=1.0e-5
  end
  @testset "t-student subcopula" begin
    co = [1. 0.5; 0.5 1.]
    srand(43)
    y = gausscopulagen(500000, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.])
    g2tsubcopula!(y, [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.], [1,2])
    @test pvalue(ExactOneSampleKSTest(y[:,1], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(y[:,2], Uniform(0,1))) > α
    @test copuladeftest(y[:,1], y[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
  end
  @testset "clayton subcopula" begin
    srand(43)
    x = gausscopulagen(500000, [1. 0.5; 0.5 1.])
    srand(43)
    v = g2clsubcopula(x[:,2], 0.5)
    @test pvalue(ExactOneSampleKSTest(v, Uniform(0,1))) > α
    srand(43)
    xcap = claytonsubcopulagen(500000, [3., 3., 3., 2., 3., 0.5])
    @test pvalue(ExactOneSampleKSTest(xcap[:,4], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xcap[:,5], Uniform(0,1))) > α
    @test lefttail(xcap[:,3], xcap[:,4], 0.001) ≈ 1/(2^(1/2)) atol=1.0e-1
    @test lefttail(xcap[:,4], xcap[:,5], 0.001) ≈ 1/(2^(1/3)) atol=1.0e-1
    @test lefttail(xcap[:,5], xcap[:,6], 0.001) ≈ 1/(2^2) atol=1.0e-1
    @test righttail(xcap[:,4], xcap[:,5], 0.999) ≈ 0 atol=1.0e-1
    @test righttail(xcap[:,3], xcap[:,4], 0.999) ≈ 0 atol=1.0e-1
    @test righttail(xcap[:,3], xcap[:,5], 0.999) ≈ 0 atol=1.0e-1
    srand(43)
    clneg = claytonsubcopulagen(500000, [-0.9, -0.9, -0.9])
    @test pvalue(ExactOneSampleKSTest(clneg[:,2], Uniform(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(clneg[:,3], Uniform(0,1))) > α
    @test lefttail(clneg[:,2], clneg[:,3], 0.001) ≈ 0
    @test righttail(clneg[:,2], clneg[:,3], 0.999) ≈ 0
  end
  @testset "test for std normal distribution of marginals of subcopdatagen" begin
    srand(43)
    xs = subcopdatagen(100000, 5, [1,2], [4,5], [1., 1., 1., 1., 3.]);
    @test pvalue(ExactOneSampleKSTest(xs[:,1], Normal(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xs[:,2], Normal(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xs[:,3], Normal(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xs[:,4], Normal(0,1))) > α
    @test pvalue(ExactOneSampleKSTest(xs[:,5],Normal(0,3))) > α
  end
end
