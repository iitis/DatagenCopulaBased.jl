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
  copulagen(copula::String, r::Matrix{Float}, θ::Union{Float64, Int})

Auxiliary function used to generate data from clayton, gumbel, frank or amh copula
parametrised by a single parameter θ given a matrix of independent [0,1] distributerd
random vectors.

"""

function copulagen(copula::String, r::Matrix{T}, θ::Union{Float64, Int}) where T <:AbstractFloat
  u = r[:,1:end-1]
  v = r[:,end]
  if copula == "clayton"
    u = -log.(u)./quantile(Gamma(1/θ, θ), v)
    return (1 + θ.*u).^(-1/θ)
  elseif copula == "amh"
    u = -log.(u)./(1+quantile(Geometric(1-θ), v))
    return (1-θ)./(exp.(u)-θ)
  elseif copula == "frank"
    u = -log.(u)./logseriesquantile(1-exp(-θ), v)
    return -log.(1+exp.(-u)*(exp(-θ)-1))/θ
  elseif copula == "gumbel"
    u = -log.(u)./levygen(θ, v)
    return exp.(-u.^(1/θ))
  end
  u
end

"""

  archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String; rev::Bool = false)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Archimedean
one parameter copula.

If reversed returns data from reversed copula.

```jldoctest
julia> srand(43);

julia> archcopulagen(10, 2, 1, "clayton")
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

function archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String;
                                                              rev::Bool = false,
                                                              cor::String = "")
  if cor == "pearson"
    θ = useρ(θ , copula)
  elseif cor == "kendall"
    θ = useτ(θ , copula)
  else
    testθ(θ, copula)
  end
  u = copulagen(copula, rand(t,n+1), θ)
  rev? 1-u: u
end


function testθ(θ::Union{Float64, Int}, copula::String)
  if copula == "gumbel"
    θ >= 1 || throw(AssertionError("generaton not supported for θ < 1"))
  else
    θ > 0 || throw(AssertionError("generaton not supported for θ ≤ 0"))
    if copula == "amh"
      1 > θ || throw(AssertionError("generator not supported for θ ≥ 1"))
    end
  end
end

function useρ(ρ::Float64, copula::String)
  0 < ρ < 1 || throw(AssertionError("correlation coeficiant must fulfill 0 < ρ < 1"))
  if copula == "amh"
    0 < ρ < 0.5 || throw(AssertionError("correlation coeficiant must fulfill 0 < ρ < 0.5"))
  end
  ρ2θ(ρ, copula)
end


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
  U = mocopula(rand(t,2^n-1), n, λ)
  reverse? 1-U: U
end

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
  ρar = [fill(ρ, (n-1))...]
  x = ordered? archcopulagen(4*n, ρar, "clayton"; cor = "pearson"): archcopulagen(4*n, n, ρ, "clayton"; cor = "pearson")
  convertmarg!(x, TDist, [[rand([2,4,5,6,7,8,9,10])] for i in 1:n])
  altersing? cor(x.*transpose(rand([-1, 1],n))): cor(x)
end
