## Elliptical copulas

"""
    gausscopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.])

Returns: t x n matrix of t realisations of multivariate data generated
using gaussian copula with Σ - correlation matrix. If the symmetric covariance
matrix is imputed, it will be converted into a correlation matrix automatically.

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

function gausscopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.])
  z = transpose(rand(MvNormal(Σ),t))
  for i in 1:size(Σ, 1)
    d = Normal(0, sqrt.(Σ[i,i]))
    z[:,i] = cdf(d, z[:,i])
  end
  z
end

"""
  tstudentcopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.], ν::Int=10)

Generates data using t-student Copula given Σ - correlation matrix, ν - degrees of freedom.
If the symmetric covariance matrix is imputed, it will be converted into a
correlation matrix automatically.

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

function tstudentcopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.], ν::Int=10)
  z = transpose(rand(MvNormal(Σ),t))
  U = rand(Chisq(ν), size(z, 1))
  for i in 1:size(Σ, 1)
    x = z[:,i].*sqrt.(ν./U)./Σ[i,i]
    z[:,i] = cdf(TDist(ν), x)
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

# Archimedean copulas
"""

  claytoncopulagen(t::Int, n::Int, θ::Float64)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Clayton
copula with parameter θ > 0.
If pearsonrho = true parameter is Pearson correlation coefficent fulfilling 0 ≥ θ > 1.
If reversed returns data from reversed Clayton copula.

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

function claytoncopulagen(t::Int, n::Int, θ::Union{Float64, Int}; pearsonrho::Bool = false,
                                                                  reverse::Bool = false)
  θ > 0 || throw(AssertionError("generaton not supported for θ ≤ 0"))
  if pearsonrho
    θ < 1 || throw(AssertionError("correlation coeficient > 1"))
    θ = θ = ρ2θ(θ, "clayton")
  end
  v = quantile(Gamma(1/θ, θ), rand(t))
  u = rand(t, n)
  u = -log.(u)./v
  u = (1 + θ.*u).^(-1/θ)
  reverse? 1-u: u
end


"""
  frankcopulagen(t::Int, n::Int, θ::Float64; pearsonrho::Bool)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Frank
copula with parameter θ > 0.
If pearsonrho = true parameter is Pearson correlation coefficent fulfilling 0 > θ > 1.

```jldoctest

julia> srand(43);

julia> @pyimport numpy.random as npr

julia> npr.seed(43);

julia> frankcopulagen(10, 3, 3.)
10×3 Array{Float64,2}:
 0.596854  0.844052    0.998418
 0.740826  0.228423    0.33928
 0.953641  0.585595    0.991504
 0.817122  0.115522    0.133532
 0.855289  0.735135    0.869024
 0.946599  0.850567    0.883339
 0.516219  0.00147225  0.245763
 0.159666  0.00928133  0.727792
 0.324894  0.386399    0.304321
 0.990873  0.96857     0.958152

```
"""
function frankcopulagen(t::Int, n::Int, θ::Union{Float64, Int}; pearsonrho::Bool = false)
  θ > 0 || throw(AssertionError("generator not supported for θ ≤ 0"))
  if pearsonrho
    θ < 1 || throw(AssertionError("correlation coeficiant must fulfill < 1"))
    θ = ρ2θ(θ, "frank")
  end
  v = npr.logseries((1-exp(-θ)), t)
  u = rand(t, n)
  u = -log.(u)./v
  -log.(1+exp.(-u)*(exp(-θ)-1))/θ
end

"""
  amhcopulagen(t::Int, n::Int, θ::Float64; pearsonrho::Bool, everse::Bool)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Ali-Mikhail-Haq
copula with parameter 0 > θ > 1.
If pearsonrho = true, parameter is Pearson correlation coefficent fulfilling 0 > θ > 0.5.
If reversed = true, returns data from reversed Ali-Mikhail-Haq copula.

```jldoctest

julia> srand(43);

julia> amhcopulagen(10, 2, 0.5)
10×2 Array{Float64,2}:
 0.494523   0.993549
 0.266095   0.417142
 0.0669154  0.960595
 0.510007   0.541976
 0.0697899  0.292847
 0.754909   0.809849
 0.0352515  0.588425
 0.32647    0.973168
 0.352815   0.247616
 0.938565   0.918152
```
"""

function amhcopulagen(t::Int, n::Int, θ::Float64; pearsonrho::Bool = false, reverse::Bool = false)
  1 > θ > 0 || throw(AssertionError("generator not supported for θ ≤ 0 or θ ≥ 1"))
  if pearsonrho
    maximum(θ) < 0.5 || throw(AssertionError("not supported for correlation ≥ 0.5"))
    θ = ρ2θ(θ, "amh")
  end
  v = 1+rand(Geometric(1-θ), t)
  u = rand(t, n)
  u = -log.(u)./v
  u = (1-θ)./(exp.(u)-θ)
  reverse? 1-u : u
end

"""
  gumbelcopulagen(t::Int, n::Int, θ::Union{Float64, Int}; pearsonrho::Bool = false,
                                                          reverse::Bool = false)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Gumbel
copula with parameter θ ≥ 1.
If pearsonrho = true, parameter is Pearson correlation coefficent fulfilling 0 ≥ θ ≥ 1.
If reversed = true, returns data from reversed Gumbel copula.

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
    θ = ρ2θ(θ, "gumbel")
  else
    θ >= 1 || throw(AssertionError("generaton not supported for θ < 1"))
  end
  v = rand(t)
  g = -sin.(pi.*v.*(1 - 1/θ))./log.(rand(t))
  v = g.^(θ-1).*sin.(pi.*v/θ)./sin.(pi.*v).^θ
  u = rand(t, n)
  u = -log.(u)./v
  u = exp.(-u.^(1/θ))
  reverse? 1-u : u
end

"""
  marshalolkincopulagen(t::Int, λ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Marshal-Olkin
copula with parameter vector λ of non-negative elements λₛ.
Number of marginals is n = ceil(Int, log(2, length(λ)-1)).
Parameters are ordered as follow:
λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]
If reversed = true, returns data from reversed Marshal-Olkin copula.

```jldoctest

julia> marshalolkincopulagen(10, [0.2, 1.2, 1.6])
10×2 Array{Float64,2}:
 0.875948   0.813807
 0.902229   0.852105
 0.386377   0.22781
 0.666248   0.381651
 0.10115    0.0283248
 0.0666898  0.00202552
 0.99636    0.994344
 0.0926391  0.95373
 0.50927    0.5957
 0.782477   0.682792
 ```
"""

function marshalolkincopulagen(t::Int, λ::Vector{Float64} = rand(7); reverse::Bool = false)
  minimum(λ) >= 0 || throw(AssertionError("all parameters must by >= 0 "))
  n = floor(Int, log(2, length(λ)+1))
  s = collect(combinations(collect(1:n)))
  l = length(s)
  U = zeros(t, n)
    for j in 1:t
      v = rand(l)
      for i in 1:n
        inds = find([i in s[k] for k in 1:l])
        ls = [-log(v[k])./(λ[k]) for k in inds]
        x = minimum(ls)
        Λ = sum(λ[inds])
        U[j,i] = exp.(-Λ*x)
      end
    end
    reverse? 1-U: U
end


# transforms univariate distributions
"""
  convertmarg!(X::Matrix, d::UnionAll, p::Vector{Vector})

Takes matrix X of realisations of size(X,2) = n dimensional random variable, with
uniform marginals numbered by i, and convert those marginals to common distribution
d with parameters p[i].
If `testunif = true` each marginal is tested for uniformity.

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

function convertmarg!(U::Matrix{T}, d::UnionAll, p::VecVec = [fill([0,1], size(U, 2))...];
                                                testunif::Bool = true) where T <: AbstractFloat
  for i = 1:size(U, 2)
    if testunif
      pvalue(ExactOneSampleKSTest(U[:,i],Uniform(0,1)))>0.0001 || throw(AssertionError("$i marg. not unif."))
    end
    @inbounds U[:,i] = quantile(d(p[i]...), U[:,i])
  end
end

  # generates covariance matrix

  """
    cormatgen(n::Int, ρ::Float64 = 0.5, ordered = false, altersing::Bool = true)

Returns symmetric correlation matrix Σ of size n x n, with reference correlation 0 < ρ < 1.
If ordered = false, Σ elements varies arround ρ, i.e. σᵢⱼ ≈ ρ+δ else they drop
as indices differences rise, i.e. σᵢⱼ ≳ σᵢₖ as |i-j| < |i-k|.
If altersing = true, some σ are positive and some negative, else ∀ᵢⱼ σᵢⱼ ≥ 0.

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
  x = ordered? claytoncopulagen(4*n, [fill(ρ, (n-1))...]; pearsonrho = true): claytoncopulagen(4*n, n, ρ; pearsonrho = true)
  convertmarg!(x, TDist, [[rand([2,4,5,6,7,8,9,10])] for i in 1:n])
  altersing? cor(x.*transpose(rand([-1, 1],n))): cor(x)
end
