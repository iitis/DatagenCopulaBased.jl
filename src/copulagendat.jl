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
    z[:,i] = cdf.(d, z[:,i])
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
    z[:,i] = cdf.(TDist(ν), x)
  end
  z
end


"""
  function frechetcopulagen(t::Int, n::Int, α::Float64)

Returns t realisation of n variate data generated from one parameter frechet multidimentional copula,
a combination of maximal copla with  weight α and independent copula with  weight 1-α

```jldoctest
julia> srand(43);

julia> frechetcopulagen(10, 2, 0.5)
10×2 Array{Float64,2}:
 0.180975  0.180975
 0.775377  0.0742681
 0.888934  0.888934
 0.924876  0.0950087
 0.408278  0.408278
 0.912603  0.740184
 0.828727  0.828727
 0.400537  0.0288987
 0.429437  0.429437
 0.955881  0.851275
```
"""


function frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64})
  0 <= α <= 1 || throw(AssertionError("generaton not supported for α ∉ [0,1]"))
  u = rand(t, n)
  p = invperm(sortperm(u[:,1]))
  l = floor(Int, t*α)
  for i in 1:n
    u[p[1:l],i] = u[p[1:l], 1]
  end
  u
end


"""
  frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64}, β::Union{Int, Float64})

Two parameters Frechet copula C = α C_{max} + β C_{min} + (1- α - β) C_{⟂}, supported
only for n == 2
"""


function frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64}, β::Union{Int, Float64})
  n == 2 || throw(AssertionError("two parameters Frechet copula supported only for n = 2"))
  bivfrechetcopulagen(t, [α], [β])
end

# Archimedean copulas
"""
  getV0(θ::Union{Float64, Int}, v::Vector{Float64}, copula::String)

Returns Vector{Float} or Vector{Int} of realisations of axiliary variable V0
used to ganarate data from 1d archimedean copula with parameter Θ, given v:
realisations of 1d variable uniformly distributed on [0,1]

```jldoctest

julia> getV0(2., [0.2, 0.4, 0.6, 0.8], "clayton")
4-element Array{Float64,1}:
 0.0641848
 0.274996
 0.708326
 1.64237
```
"""

function getV0(θ::Union{Float64, Int}, v::Vector{Float64}, copula::String)
  if copula == "clayton"
    return quantile.(Gamma(1/θ, θ), v)
  elseif copula == "amh"
    return 1+quantile.(Geometric(1-θ), v)
  elseif copula == "frank"
    return logseriesquantile(1-exp(-θ), v)
  elseif copula == "gumbel"
    return levygen(θ, v)
  end
  v
end

"""
  phi(u::Matrix{Float64}, θ::Union{Float64, Int}, copula::String)

Given a matrix t realisations of n variate data ℜᵗⁿ ∋ u = -log(rand(t,n))./V0
returns it transformed through an inverse generator of archimedean copula. Output
is distributed uniformly on [0,1]ⁿ

```jldoctest

julia> julia> phi([0.2 0.6; 0.4 0.8], 2., "clayton")
2×2 Array{Float64,2}:
 0.845154  0.6742
 0.745356  0.620174
```
"""

function phi(u::Matrix{Float64}, θ::Union{Float64, Int}, copula::String)
  if copula == "clayton"
    return (1 + θ.*u).^(-1/θ)
  elseif copula == "amh"
    return (1-θ)./(exp.(u)-θ)
  elseif copula == "frank"
    return -log.(1+exp.(-u)*(exp(-θ)-1))/θ
  elseif copula == "gumbel"
    return exp.(-u.^(1/θ))
  end
  u
end

"""
  copulagen(copula::String, r::Matrix{Float}, θ::Union{Float64, Int})

Auxiliary function used to generate data from archimedean copula (clayton, gumbel, frank or amh)
parametrised by a single parameter θ given a matrix of independent [0,1] distributerd
random vectors.

```jldoctest
  julia> copulagen("clayton", [0.2 0.6 0.9; 0.4 0.5 0.8], 2.)
  2×2 Array{Float64,2}:
   0.675778  0.851993
   0.687482  0.736394
```

"""

function copulagen(copula::String, r::Matrix{T}, θ::Union{Float64, Int}) where T <:AbstractFloat
  u = -log.(r[:,1:end-1])./getV0(θ, r[:,end], copula)
  phi(u, θ, copula)
end

"""

  archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String; rev::Bool = false)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Archimedean
one parameter copula.

Following copula families are supported: clayton, frank, gumbel and amh --
Ali-Mikhail-Haq.

If rev == true, reverse the copula output i.e. u → 1-u (we call it reversed copula).
It cor == pearson, kendall, uses correlation coeficient as a parameter

```jldoctest
julia> srand(43);

julia> archcopulagen(10, 2, 1, "clayton")
10×2 Array{Float64,2}:
 0.770331  0.932834
 0.472847  0.0806845
 0.970749  0.653029
 0.622159  0.0518025
 0.402461  0.228549
 0.946375  0.842883
 0.809076  0.129038
 0.747983  0.433829
 0.374341  0.437269
 0.973066  0.910103

 ```
"""

function archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String;
                                                              rev::Bool = false,
                                                              cor::String = "")
  cor in ["pearson", "kendall", ""] || throw(AssertionError("$(cor) correlation not supported"))
  copula in ["clayton", "amh", "frank", "gumbel"] || throw(AssertionError("$(copula) copula is not supported"))
  if *(n == 2)*((copula != "gumbel")*(θ < 0) | (copula == "amh")*(θ in [0,1]))
    return bivariatecopgen(t, [θ], copula; rev=rev, cor=cor)
  elseif cor == "pearson"
    θ = useρ(θ , copula)
  elseif cor == "kendall"
    θ = useτ(θ , copula)
  else
    testθ(θ, copula)
  end
  u = copulagen(copula, rand(t,n+1), θ)
  rev? 1-u: u
end

"""
  function testθ(θ::Union{Float64, Int}, copula::String)

Tests the parameter θ value for archimedean copula, returns void
"""

function testθ(θ::Union{Float64, Int}, copula::String)
  if copula == "gumbel"
    θ >= 1 || throw(AssertionError("generaton not supported for θ < 1"))
  elseif copula == "amh"
    1 > θ > 0 || throw(AssertionError("amh multiv. copula supported only for 0 < θ < 1"))
  else
    θ > 0 || throw(AssertionError("generaton not supported for θ ≤ 0"))
  end
end

"""
  useρ(ρ::Float64, copula::String)

Tests the available pearson correlation for archimedean copula, returns Float,
corresponding copula parameter θ.
```jldoctest

julia> useρ(0.75, "gumbel")
2.294053859606698
```

"""

function useρ(ρ::Float64, copula::String)
  0 < ρ < 1 || throw(AssertionError("correlation coeficiant must fulfill 0 < ρ < 1"))
  if copula == "amh"
    0 < ρ < 0.5 || throw(AssertionError("correlation coeficiant must fulfill 0 < ρ < 0.5"))
  end
  ρ2θ(ρ, copula)
end

"""
  useτ(ρ::Float64, copula::String)

Tests the available kendall's correlation for archimedean copula, returns Float,
corresponding copula parameter θ.

```jldoctest

julia> useτ(0.5, "clayton")
2.0
```
"""

function useτ(τ::Float64, copula::String)
  0 < τ < 1 || throw(AssertionError("correlation coeficiant must fulfill 0 < τ < 1"))
  if copula == "amh"
    0 < τ < 1/3 || throw(AssertionError("correlation coeficiant must fulfill 0 < τ < 1/3"))
  end
  τ2θ(τ, copula)
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

julia> srand(43)

julia> marshalolkincopulagen(10, [0.2, 1.2, 1.6])
10×2 Array{Float64,2}:
 0.99636   0.994344
 0.167268  0.0619408
 0.977418  0.965093
 0.495167  0.0247053
 0.410336  0.250159
 0.778989  0.678064
 0.50927   0.350059
 0.925875  0.887095
 0.353646  0.219006
 0.782477  0.686799
 ```
"""


function marshalolkincopulagen(t::Int, λ::Vector{Float64} = rand(7); reverse::Bool = false)
  minimum(λ) >= 0 || throw(AssertionError("all parameters must by >= 0 "))
  n = floor(Int, log(2, length(λ)+1))
  U = mocopula(rand(t,2^n-1), n, λ)
  reverse? 1-U: U
end


"""
  mocopula(u::Matrix{Float64}, n::Int, λ::Vector{Float64})

  Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Marshal-Olkin
  copula with parameter vector λ of non-negative elements λₛ, given [0,1]ᵗˣˡ ∋ u, where
  l = 2ⁿ-1

```jldoctest

  julia> mocopula([0.2 0.3 0.4; 0.3 0.4 0.6; 0.4 0.5 0.7], 2, [1., 1.5, 2.])
  3×2 Array{Float64,2}:
   0.252982  0.201189
   0.464758  0.409039
   0.585662  0.5357

```
"""

function mocopula(u::Matrix{Float64}, n::Int, λ::Vector{Float64})
  s = collect(combinations(1:n))
  t,l = size(u)
  U = zeros(t, n)
    for j in 1:t
      for i in 1:n
        inds = find([i in s[k] for k in 1:l])
        x = minimum([-log(u[j,k])./(λ[k]) for k in inds])
        Λ = sum(λ[inds])
        U[j,i] = exp.(-Λ*x)
      end
    end
    U
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
    @inbounds U[:,i] = quantile.(d(p[i]...), U[:,i])
  end
end

  # generates covariance matrix

  """
    cormatgen(n::Int, ρ::Float64 = 0.5, ordered::Bool = false, altersing::Bool = true)

Returns symmetric correlation matrix Σ of size n x n, with reference correlation 0 < ρ < 1.
If ordered = false, Σ elements varies arround ρ, i.e. σᵢⱼ ≈ ρ+δ else they drop
as indices differences rise, i.e. σᵢⱼ ≳ σᵢₖ as |i-j| < |i-k|.
If altersing = true, some σ are positive and some negative, else ∀ᵢⱼ σᵢⱼ ≥ 0.

```jldoctest
julia> srand(43);

julia> cormatgen(4)
4×4 Array{Float64,2}:
  1.0        0.574741  -0.789649  -0.654538
  0.574741   1.0       -0.717196  -0.610049
 -0.789649  -0.717196   1.0        0.703387
 -0.654538  -0.610049   0.703387   1.0
```
"""

function cormatgen(n::Int, ρ::Float64 = 0.5, ordered::Bool = false, altersing::Bool = true)
  1 > ρ > 0 || throw(AssertionError("only 1 > ρ > 0 supported"))
  x = zeros(4*n,n)
  if ordered
    x = bivariatecopgen(4*n, [fill(ρ, (n-1))...], "clayton"; cor = "pearson")
  else
    x = archcopulagen(4*n, n, ρ, "clayton"; cor = "pearson")
  end
  convertmarg!(x, TDist, [[rand([2,4,5,6,7,8,9,10])] for i in 1:n])
  altersing? cor(x.*transpose(rand([-1, 1],n))): cor(x)
end
