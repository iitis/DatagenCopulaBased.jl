α = 0.025

@testset "heplers" begin
  @testset "axiliary functions" begin
    srand(43)
    @test cormatgen(2) ≈ [1.0 0.903212; 0.903212 1.0] atol=1.0e-5
  end
  @testset "tail dependencies" begin
    v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
    v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
    @test lefttail(v1, v2, 0.1) ≈ 0.5
    @test righttail(v1, v2, 0.9) ≈ 0.5
  end
  @testset "correlations vs parameter" begin
    @test ρ2θ(0.3090169943749474, "clayton") ≈ 0.5
    @test ρ2θ(0.08694, "frank") ≈ 0.5 atol=1.0e-3
    @test ρ2θ(0.5, "gumbel") ≈ 1.5
    @test AMHθ(0.2) ≈ 0.5 atol=1.0e-1
  end
  @testset "getting parameter" begin
    @test τ2λ([0.405154], [4.2, 1.2]) ≈ [4.2, 1.2, 3.7] atol=1.0e-1
    @test τ2θ(0.6, "frank") ≈ 7.929642284264058
    @test τ2θ(0.5, "gumbel") ≈ 2.
    @test τ2θ(1/3, "clayton") ≈ 1.
    @test τ2θ(1/4, "amh") ≈ 0.8384520912688538
  end
  @testset "copula gen" begin
  c = copulagen("clayton", [0.2 0.4 0.8; 0.2 0.8 0.6; 0.3 0.9 0.6], 1.)
  @test c ≈ [0.5 0.637217; 0.362783 0.804163; 0.432159 0.896872] atol=1.0e-5
  end
end

@testset "data generators" begin
  @testset "logseries dist" begin
    @test logseriescdf(0.01)[1:3] ≈ [0.0, 0.994992, 0.999967] atol=1.0e-5
    @test logseriesquantile(0.9, [0.25, 0.5, 0.75]) == [1, 2, 5]
    srand(43)
    v = logseriesquantile(0.4, rand(1000000))
    @test mean(v) ≈ 1.304 atol=1.0e-2
    @test std(v) ≈ 0.687 atol=1.0e-2
    @test skewness(v) ≈ 3.1 atol=1.0e-2
    @test kurtosis(v) ≈ 13.5 atol=1.0
  end
end

@testset "transform marginals" begin
  x1 = [0.2 0.4; 0.4 0.6; 0.6 0.8]
  convertmarg!(x1, TDist, [[10],[10]])
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

@testset "gaussian copula" begin
  srand(43)
  x = gausscopulagen(500000, [1. 0.5; 0.5 1.])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test lefttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
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
  @test lefttail(xt[:,1], xt[:,2]) ≈ λ atol=1.0e-1
  @test righttail(xt[:,1], xt[:,2]) ≈ λ atol=1.0e-1
  convertmarg!(xt, Normal)
  @test cor(xt) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
end
@testset "product copula" begin
  srand(43)
  x = productcopula(500000, 3);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test lefttail(x[:,1], x[:,3]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
end
@testset "gumbel copula" begin
  srand(43)
  x = gumbelcopulagen(500000, 3, 2.);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test righttail(x[:,1], x[:,2]) ≈ 0.5858 atol=1.0e-1
  @test righttail(x[:,1], x[:,2]) ≈ 0.5858 atol=1.0e-1
  @test lefttail(x[:,1], x[:,2]) ≈ 0. atol=1.0e-1
  @test lefttail(x[:,1], x[:,3]) ≈ 0. atol=1.0e-1
  @test corkendall(x) ≈ [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.] atol=1.0e-3
  srand(44)
  x = gumbelcopulagen(500000, 3, 1.5; reverse = true);
  @test lefttail(x[:,1], x[:,2]) ≈ 0.4126 atol=1.0e-1
  @test righttail(x[:,1], x[:,3]) ≈ 0. atol=1.0e-1
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  srand(43)
  x = gumbelcopulagen(500000, 3, 0.5; pearsonrho = true)
  convertmarg!(x, Normal)
  @test cor(x) ≈ [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.] atol=1.0e-1
end
@testset "clayton copula" begin
  srand(43)
  x = claytoncopulagen(500000, 3, 1);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test lefttail(x[:,1], x[:,2]) ≈ 0.5 atol=1.0e-1
  @test lefttail(x[:,1], x[:,3]) ≈ 0.5 atol=1.0e-1
  @test righttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
  @test corkendall(x) ≈ [1. 1/3 1/3; 1/3 1. 1/3; 1/3 1/3 1.] atol=1.0e-2
  srand(43)
  xic = claytoncopulagen(500000, 3, 1; reverse = true);
  @test pvalue(ExactOneSampleKSTest(xic[:,1], Uniform(0,1))) > α
  @test lefttail(xic[:,1], xic[:,3]) ≈ 0 atol=1.0e-1
  @test righttail(xic[:,1], xic[:,2]) ≈ 0.5 atol=1.0e-1
  srand(43)
  x = claytoncopulagen(500000, 3, 0.5; pearsonrho = true)
  convertmarg!(x, Normal)
  @test cor(x) ≈ [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.] atol=1.0e-2
end
@testset "frank copula" begin
  srand(43)
  x = frankcopulagen(500000, 5, 0.8)
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,5], Uniform(0,1))) > α
  @test lefttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
  @test lefttail(x[:,2], x[:,3]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,3], x[:,2]) ≈ 0 atol=1.0e-1
  convertmarg!(x, Normal)
  @test cor(x)[1:3, 1:3] ≈ [1. 0.138 0.138; 0.138 1. 0.138; 0.138 0.138 1.] atol=1.0e-1
end
@testset "Ali-Mikhail-Haq copula" begin
  srand(43)
  x = amhcopulagen(500000, 4, 0.8)
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,4], Uniform(0,1))) > α
  @test lefttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
  @test lefttail(x[:,2], x[:,3]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,1], x[:,2]) ≈ 0 atol=1.0e-1
  @test righttail(x[:,2], x[:,3]) ≈ 0 atol=1.0e-1
  @test corkendall(x)[1:2, 1:2] ≈ [1. 0.23373; 0.23373 1.] atol=1.0e-3
end
@testset "Marhall-Olkin copula" begin
  srand(43)
  x = marshalolkincopulagen(100000, [1.1, 0.2, 0.6])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  a2 = 0.6/0.8
  a1 = 0.6/1.7
  @test corkendall(x)[1,2]≈ a1*a2/(a1+a2-a1*a2) atol=1.0e-3
  @test righttail(x[:,1], x[:,2]) ≈ a1 atol=1.0e-1
  srand(42)
  x = marshalolkincopulagen(100000, [1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
end
