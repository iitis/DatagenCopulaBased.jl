
# generated data using copulas
"""

  claytoncopulagen(t::Int, n::Int, θ::Float64)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Clayton
copula with parameter θ > 0. If pearsonrho = true parameter 0 >= θ >= 1 is taken as a
Pearson correlation coefficent. If reversed returns data from reversed Clayton copula.

```jldoctest
julia> srand(43);

julia> claytoncopulagen(10, 2, 1)
10×2 Array{Float64,2}:
  0.325965  0.984025
  0.364814  0.484407
  0.514236  0.990846
  0.523757  0.55038
  0.204864  0.398564
  0.890124  0.916516
  0.247198  0.746308
  0.126174  0.882004
  0.462986  0.377842
  0.950937  0.934698
 ```
"""

function claytoncopulagen(t::Int, n::Int = 2, θ::Union{Float64, Int} = 1.0;
                                            pearsonrho::Bool = false, reverse::Bool = false)
  θ > 0 || throw(AssertionError("generaton not supported for θ <= 0"))
  if pearsonrho
    θ <= 1 || throw(AssertionError("correlation coeficient > 1"))
    θ = claytonθ(θ)
  end
  qamma_dist = Gamma(1/θ, θ)
  x = rand(t)
  u = rand(t, n)
  u = -log.(u)./quantile(qamma_dist, x)
  u = (1 + θ.*u).^(-1/θ)
  reverse? 1-u: u
end

"""
  gumbelcopulagen(t::Int, n::Int, θ::Union{Float64, Int}; pearsonrho::Bool = false,
                                                          reverse::Bool = false)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Gumbel
copula with parameter θ > 0. If pearsonrho = true parameter 0 >= θ >= 1 is taken as a
Pearson correlation coefficent. If reversed returns data from reversed Gumbel copula.

```jldoctest
julia> srand(43);

julia> gumbelcopulagen(10, 3, 3.5)
10×3 Array{Float64,2}:
 0.550199  0.574653   0.486977
 0.352515  0.0621575  0.072297
 0.31809   0.112819   0.375482
 0.652536  0.691707   0.645668
 0.988459  0.989946   0.986297
 0.731589  0.532971   0.678277
 0.62426   0.625661   0.851237
 0.335811  0.117504   0.329193
 0.504036  0.672722   0.561857
 0.326098  0.459547   0.117946
 ```
 """

function gumbelcopulagen(t::Int, n::Int, θ::Union{Float64, Int}; pearsonrho::Bool = false,
                                                                 reverse::Bool = false)
  if pearsonrho
    0 < θ < 1 || throw(AssertionError("generaton not supported for correlation <= 0 v >= 1"))
    θ = gumbelθ(θ)
  else
    θ > 1 || throw(AssertionError("generaton not supported for θ <= 1"))
  end
  u = rand(t, n)
  v = rand(t)
  g = sin.(pi.*v.*(1 - 1/θ))./(-log.(rand(t)))
  g = (sin.(pi.*v/θ)./g).^(1/θ).*g./sin.(pi.*v)
  u = exp.(-(-log.(u)).^(1/θ)./g)
  reverse? 1-u : u
end

"""
  tstudentcopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], nu::Int=10)

Generates data using t-student Copula given a correlation matrix cormat and nu degrees of freedom

```jldoctest
julia> srand(43);

julia> tstudentcopulagen(10)
10×2 Array{Float64,2}:
 0.658199  0.937148
 0.718244  0.92602
 0.809521  0.0980325
 0.263068  0.222589
 0.187187  0.971109
 0.245373  0.346428
 0.452336  0.524498
 0.57113   0.272525
 0.498443  0.48082
 0.113788  0.633349
 ```
"""

function tstudentcopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]],
                                   nu::Int=10)
  z = transpose(rand(MvNormal(cormat),t))
  d = Chisq(nu)
  U = rand(d, size(z, 1))
  p = TDist(nu)
  for i in 1:size(cormat, 1)
    z[:,i] = cdf(p, z[:,i].*sqrt.(nu./U)./cormat[i,i])
  end
  z
end

"""
    gausscopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]])

Returns: t x n matrix of t realisations of multivariate data generated
using gaussian copula with correlation matrix - cormat

```jldoctest

julia> srand(43);

julia> gausscopulagen(10)
10×2 Array{Float64,2}:
 0.589188  0.815308
 0.708285  0.924962
 0.747341  0.156994
 0.227634  0.183116
 0.227575  0.957376
 0.271558  0.364803
 0.445691  0.52792
 0.585362  0.23135
 0.498593  0.48266
 0.190283  0.594451
```
"""

function gausscopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]])
  z = transpose(rand(MvNormal(cormat),t))
  for i in 1:size(cormat, 1)
    d = Normal(0, sqrt.(cormat[i,i]))
    z[:,i] = cdf(d, z[:,i])
  end
  z
end

"""
  productcopula(t::Int, n::Int)

Returns t realisation of n variate data generated from produce (independent) copula

```jldoctest
julia> srand(43);

julia> productcopula(10, 2)
10×2 Array{Float64,2}:
 0.180975  0.661781
 0.775377  0.0742681
 0.888934  0.125437
 0.924876  0.0950087
 0.408278  0.130474
 0.912603  0.740184
 0.828727  0.00463791
 0.400537  0.0288987
 0.429437  0.521601
 0.955881  0.851275
```
"""
productcopula(t::Int, n::Int) = rand(t,n)

# transforms univariate distributions
"""
  convertmarg!(X::Matrix, dist, p::Vector{Vector})

Takes t x n matrix of t realisations of n variate data with uniform marginals and
convert marginals on [0,1] and convert i th marginals to those with distribution dist
and parameters p[i]

```jldoctest
julia> srand(43);

julia> x = rand(10,2);

julia> convertmarg!(x, Normal, [[0, 1],[0, 10]])

julia> x
10×2 Array{Float64,2}:
 -0.911655    4.17328
  0.756673  -14.4472
  1.22088   -11.4823
  1.43866   -13.1053
 -0.231978  -11.2415
  1.35696     6.43914
  0.949145  -26.0172
 -0.251957  -18.9723
 -0.177808    0.54172
  1.70477    10.4192
```
"""
VecVec = Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}


function convertmarg!(U::Matrix{T}, dist, p::VecVec = [fill([0,1], size(U, 2))...];
                                          testunif::Bool = true) where T <: AbstractFloat
  d = Uniform(0,1)
  for i = 1:size(U, 2)
    pw = pvalue(ExactOneSampleKSTest(U[:,i],d))
    testunif? pw > 0.0001 || throw(AssertionError("$i marginal not uniform")): ()
    @inbounds U[:,i] = quantile(dist(p[i]...), U[:,i])
  end
end

  # generates covariance matrix

  """
    cormatgen(n::Int, rho::Float64 = 0.5, ordered = false, altersing::Bool = true)

Returns symmetric correlation matrix of size `n x n`, with reference correlation 0 < rho < 1.
If ordered = false, matrix correlation matrix elements varies arround rho, else it drops
as a distance between marginal variables risis. If altersing = true some elements are positive
and some negative, else all pelements are postive.

```jldoctest
julia> srand(43);

julia> cormatgen(4)
4×4 Array{Float64,2}:
  1.0        0.566747  -0.34848   -0.413496
  0.566747   1.0       -0.496956  -0.575852
 -0.34848   -0.496956   1.0        0.612688
 -0.413496  -0.575852   0.612688   1.0
```
"""

  function cormatgen(n::Int, ρ::Float64 = 0.5, ordered::Bool = false, altersing::Bool = true)
    1 > ρ > 0 || throw(AssertionError("only 1 > ρ > 0 supported"))
    ρ = ordered? [fill(ρ, (n-1))...]: ρ
    x = claytoncopulagen(4*n, n, ρ; pearsonrho = true)
    convertmarg!(x, TDist, [[rand([2,4,5,6,7,8,9,10])] for i in 1:n])
    altersing? cor(x): cor(x.*transpose(rand([-1, 1],n)))
  end
