
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
    @test clcopulagen(2,2) ≈ [0.629041  0.182246; 0.950303  0.942292] atol=1.0e-5
    srand(43)
    @test tcopulagen([[1. 0.5];[0.5 1.]], 2) ≈ [0.581625 0.792144; 0.76935 0.968669] atol=1.0e-5
    srand(43)
    @test gcopulagen([[1. 0.5];[0.5 1.]], 2) ≈ [0.589188 0.815308; 0.708285 0.924962] atol=1.0e-5
  end
  @testset "transform marginals" begin
    x = [0.2 0.4; 0.4 0.6; 0.6 0.8]
    x1 = [0.2 0.4; 0.4 0.6; 0.6 0.8]
    convertmarg!(x, Normal, [[0., 1.],[0., 1.]])
    convertmarg!(x1, TDist, [[10],[10]])
    @test x ≈ [-0.841621 -0.253347; -0.253347 0.253347; 0.253347 0.841621] atol=1.0e-5
    @test x1 ≈ [-0.879058  -0.260185; -0.260185 0.260185; 0.260185 0.879058] atol=1.0e-5
  end
end

@testset "subcopulas" begin
  srand(43)
  cov = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.]
  x = gcopulagen(cov, 3)
  x1 = copy(x)
  g2tsubcopula!(x1, cov, [1,2])
  @test x1 ≈ [0.558652  0.719921  0.794493; 0.935573  0.922409  0.345177; 0.217512  0.174138  0.123049] atol=1.0e-5
  v = g2clsubcopula(x[:,2], 0.5)
  @test v ≈ [0.31555, 0.846364, 0.0132052] atol=1.0e-5
end

@testset "helpers" begin
  v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
  v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
  @test lefttail(v1, v2, 0.1) ≈ 0.5
  @test righttail(v1, v2, 0.9) ≈ 0.5
end



cov = [1. 0.5; 0.5 1.]
srand(40)
x = gcopulagen(cov, 200000)
v = g2clsubcopula(x[:,2], cov[1,2])
y = copy(x)
ν = 6
g2tsubcopula!(y, cov, [1,2])
xt = tcopulagen(cov, 200000, ν);
xc = clcopulagen(200000, 2);
α = 0.05
@testset "tests for uniform distribution" begin
  d = Uniform(0,1)
  @test pvalue(ExactOneSampleKSTest(x[:,1],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2],d)) > α
  @test pvalue(ExactOneSampleKSTest(y[:,1],d)) > α
  @test pvalue(ExactOneSampleKSTest(y[:,2],d)) > α
  @test pvalue(ExactOneSampleKSTest(v,d)) > α
  @test pvalue(ExactOneSampleKSTest(xt[:,1],d)) > α
  @test pvalue(ExactOneSampleKSTest(xt[:,2],d)) > α
  @test pvalue(ExactOneSampleKSTest(xc[:,1],d)) > α
  @test pvalue(ExactOneSampleKSTest(xc[:,2],d)) > α
end
@testset "copula def" begin
  @test copuladeftest(x[:,1], x[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
  @test copuladeftest(y[:,1], y[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
  @test copuladeftest(x[:,1], v, [0.7, 0.8], [0.2, 0.7]) > 0
  @test copuladeftest(xt[:,1], xt[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
  @test copuladeftest(xc[:,1], xc[:,2], [0.5, 0.9], [0.2, 0.7]) > 0
end
@testset "tail test" begin
  @test lefttail(x[:,1], x[:,2], 0.001) ≈ 0 atol=1.0e-1
  @test righttail(x[:,1], x[:,2], 0.999) ≈ 0 atol=1.0e-1
  @test lefttail(xc[:,1], xc[:,2], 0.001) ≈ 0.5 atol=1.0e-1
  @test righttail(xc[:,1], xc[:,2], 0.999) ≈ 0 atol=1.0e-1
  d = TDist(ν+1)
  rho = cov[1,2]
  λ = 2*pdf(d, -sqrt.((ν+1)*(1-rho)/(1+rho)))
  @test lefttail(xt[:,1], xt[:,2], 0.001) ≈ λ atol=1.0e-1
  @test righttail(xt[:,1], xt[:,2], 0.999) ≈ λ atol=1.0e-1
end
@testset "test for std normal distribution of marginals of subcopdatagen" begin
  x = subcopdatagen([1,2], [4,5], 100000, 5);
  d = Normal(0,1)
  @test pvalue(ExactOneSampleKSTest(x[:,1],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5],d)) > α
  x = subcopdatagen([1,2], [], 100000, 3, [3., 3., 3.]);
  d = Normal(0,3)
  @test pvalue(ExactOneSampleKSTest(x[:,1],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2],d)) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3],d)) > α
end
