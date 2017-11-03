α = 0.025

@testset "heplers" begin
  @testset "tail dependencies" begin
    v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
    v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
    @test tail(v1, v2,  "l", 0.1) ≈ 0.5
    @test tail(v1, v2, "r", 0.1) ≈ 0.5
  end
  @testset "parameter formo kendall's cor" begin
    @test Debye(0.5, 1) ≈ 0.8819271567906056
    @test τ2λ([0.4], [4.2, 1.2]) ≈ [4.2, 1.2, 3.6]
    @test τ2θ(0.6, "frank") ≈ 7.929642284264058
    @test frankτ2θ(0.6) ≈ 7.929642284264058
    @test τ2θ(0.5, "gumbel") ≈ 2.
    @test τ2θ(1/3, "clayton") ≈ 1.
    @test τ2θ(1/4, "amh") ≈ 0.8384520912688538
    @test AMHτ2θ(1/4) ≈ 0.8384520912688538
  end
  @testset "parameter from pearson cor" begin
    @test dilog(0.5) ≈ 0.5822405264650125
    @test ρ2θ(1/3, "clayton") ≈ 0.5799054034685394
    @test ρ2θ(1/3, "frank") ≈ 2.1164969117225363
    @test ρ2θ(0.5, "gumbel") ≈ 1.5390534821085031
    @test ρ2θ(0.2, "amh") ≈ 0.5168580913147318
    @test frankθ(1/3) ≈ 2.1164969117225363
    @test AMHθ(0.2) ≈ 0.5168580913147318
  end
end



@testset "transform marginals" begin
  x1 = [0.2 0.4; 0.4 0.6; 0.6 0.8]
  convertmarg!(x1, TDist, [[10],[10]])
  @test x1 ≈ [-0.879058  -0.260185; -0.260185 0.260185; 0.260185 0.879058] atol=1.0e-5
  srand(43)
  x = rand(10000, 2)
  convertmarg!(x, Normal, [[0., 2.],[0., 3.]])
  @test pvalue(ExactOneSampleKSTest(x[:,1],Normal(0,2))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2],Normal(0,3))) > α
  srand(43)
  @test_throws AssertionError convertmarg!(randn(1000, 2), Normal)
end

@testset "generate corelation matrix" begin
  srand(43)
  @test cormatgen(2) ≈ [1.0 0.911683; 0.911683 1.0] atol=1.0e-5
  srand(44)
  @test cormatgen(3, 0.8, true, false) ≈ [ 1.0 0.586417 0.461157; 0.586417 1.0 0.732091; 0.461157 0.732091 1.0] atol=1.0e-5
  srand(44)
  @test cormatgen(3, 0.8, true, true) ≈ [ 1.0 0.586417 -0.461157; 0.586417 1.0 -0.732091; -0.461157 -0.732091 1.0] atol=1.0e-5
end

@testset "gaussian copula" begin
  srand(43)
  x = gausscopulagen(500000, [1. 0.5; 0.5 1.])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0
end
@testset "t-student copula" begin
  ν = 10
  rho = 0.5
  λ = 2*pdf(TDist(ν+1), -sqrt.((ν+1)*(1-rho)/(1+rho)))
  srand(43)
  xt = tstudentcopulagen(500000, [1. rho; rho 1.], ν);
  @test pvalue(ExactOneSampleKSTest(xt[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(xt[:,2], Uniform(0,1))) > α
  @test tail(xt[:,1], xt[:,2], "l") ≈ λ atol=1.0e-1
  @test tail(xt[:,1], xt[:,2], "r") ≈ λ atol=1.0e-1
  convertmarg!(xt, Normal)
  @test cor(xt) ≈ [1. rho; rho 1.] atol=1.0e-2
end
@testset "frechet copula" begin
  srand(43)
  x = frechetcopulagen(500000, 3, 0.3);
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l") ≈ 0.3 atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r") ≈ 0.3 atol=1.0e-1
  @test cor(x) ≈ [1. 0.3 0.3; 0.3 1. 0.3; 0.3 0.3 1.] atol=1.0e-2
  x = frechetcopulagen(500000, 3, 1.)
  ret = (x[:, 1] .<  0.2).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
  @test length(find(ret))/size(x,1) ≈ 0.2 atol=1.0e-3
  ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.3).*(x[:, 3] .<  0.5)
  @test length(find(ret))/size(x,1) ≈ 0.3 atol=1.0e-3
  ret = (x[:, 1] .<  0.8).*(x[:, 2] .<  0.9).*(x[:, 3] .<  0.5)
  @test length(find(ret))/size(x,1) ≈ 0.5 atol=1.0e-3
  srand(43)
  x = frechetcopulagen(500000, 2, .0)
  @test cor(x) ≈ [1. 0.; 0. 1.] atol=1.0e-2
end

@testset "archimedean copulas axiliary functions" begin
  @test getV0(2., [0.2, 0.4, 0.6, 0.8], "clayton") ≈ [0.0641848, 0.274996, 0.708326, 1.64237] atol=1.0e-4
  @test phi([0.2 0.6; 0.4 0.8], 2., "clayton") ≈ [0.845154  0.6742; 0.745356  0.620174] atol=1.0e-4
  c = copulagen("clayton", [0.2 0.4 0.8; 0.2 0.8 0.6; 0.3 0.9 0.6], 1.)
  @test c ≈ [0.5 0.637217; 0.362783 0.804163; 0.432159 0.896872] atol=1.0e-5
  @test useτ(0.5, "clayton") == 2.
  @test useρ(0.75, "gumbel") ≈ 2.294053859606698
end

@testset "archimedean copulas exceptions" begin
  @test_throws(AssertionError, testθ(0.5, "gumbel"))
  @test_throws(AssertionError, useρ(0.6, "amh"))
  @test_throws(AssertionError, useτ(0.45, "amh"))
  @test_throws(AssertionError, archcopulagen(100000, 4, -0.6, "frank"))
end

@testset "gumbel copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 2., "gumbel");
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
  @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.
  @test corkendall(x) ≈ [1. 1/2 1/2; 1/2 1. 1/2; 1/2 1/2 1.] atol=1.0e-2
  srand(43)
  x = archcopulagen(500000, 2, 1.5, "gumbel"; rev = true)
  @test tail(x[:,1], x[:,2], "l") ≈ 2-2^(1/1.5) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.00001) ≈ 0.
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  srand(43)
  x = archcopulagen(500000, 2, 0.5, "gumbel"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-4
  srand(43)
  x = archcopulagen(500000, 2, 0.5, "gumbel"; cor = "pearson")
  @test cor(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-2
end
@testset "clayton copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 1., "clayton");
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
  @test tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x) ≈ [1. 1/3 1/3; 1/3 1. 1/3; 1/3 1/3 1.] atol=1.0e-2
  srand(43)
  x = archcopulagen(500000, 2, 0.5, "clayton"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.5; 0.5 1.] atol=1.0e-3
  srand(43)
  x = archcopulagen(500000, 2, -0.9, "clayton")
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test corkendall(x)[1,2] ≈ -0.9/(2-0.9) atol=1.0e-3
end
@testset "frank copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 0.8, "frank")
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,2], x[:,3], "r", 0.0001) ≈ 0
  srand(43)
  x = archcopulagen(500000, 2, 0.2, "frank"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.2; 0.2 1.] atol=1.0e-3
end
@testset "Ali-Mikhail-Haq copula" begin
  srand(43)
  x = archcopulagen(500000, 3, 0.8, "amh")
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
  @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
  @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
  @test corkendall(x)[1:2, 1:2] ≈ [1. 0.23373; 0.23373 1.] atol=1.0e-3
  x = archcopulagen(500000, 2, 0.25, "amh"; cor = "kendall")
  @test corkendall(x) ≈ [1. 0.25; 0.25 1.] atol=1.0e-3
end
@testset "Marhall-Olkin copula" begin
  m = [0.252982 0.201189;  0.464758 0.409039; 0.585662 0.5357]
  @test mocopula([0.2 0.3 0.4; 0.3 0.4 0.6; 0.4 0.5 0.7], 2, [1., 1.5, 2.]) ≈ m atol=1.0e-4
  srand(43)
  x = marshalolkincopulagen(100000, [1.1, 0.2, 0.6])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  a2 = 0.6/0.8
  a1 = 0.6/1.7
  @test corkendall(x)[1,2]≈ a1*a2/(a1+a2-a1*a2) atol=1.0e-3
  @test tail(x[:,1], x[:,2], "r") ≈ a1 atol=1.0e-1
  srand(42)
  x = marshalolkincopulagen(100000, [1.1, 0.2, 2.1, 0.6, 0.5, 3.2, 7.1, 2.1])
  @test pvalue(ExactOneSampleKSTest(x[:,1], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,2], Uniform(0,1))) > α
  @test pvalue(ExactOneSampleKSTest(x[:,3], Uniform(0,1))) > α
end
