### Following R. Nelsen 'An Introduction to Copulas', Springer Science & Business Media, 1999 - 216,
### for bivariate Archimedean copulas `C(u₁,u₂)` data can be generated as follow:
### * draw `u₁ = rand()`,
### * define `w = ∂C(u₁, u₂)\∂u₁` and inverse `u₂ = f(w, u₁)`,
### * draw  `w = rand()`
### * return a pair u₁, u₂.

### This method is applied here for Clayton, Frank and Ali-Mikhail-Haq copula. If we use
### this method recursively, we can get `n`-variate data with uniform marginals on
### `[0,1]`, where each neighbour pair
### of marginals `uᵢ uⱼ` for `j = i+1` are draw form a bivariate subcopula with
### parameter `θᵢ`, the only condition for `θᵢ`
### is such as for a corresponding bivariate copula.

"""
  rand2cop(u1::Real, θ::Real, copula::String, w::Real)

Returns Real  generated using copula::String given u1 and w from uniform distribution
and copula parameter θ.
"""
function rand2cop(u1::T, θ::T, copula::String, w::T) where T <: Real
  copula in ["clayton", "amh", "frank"] || throw(AssertionError("$(copula) copula is not supported"))
  if copula == "clayton"
    return (u1^(-θ)*(w^(-θ/(1+θ)) -1) +1)^(-1/θ)
  elseif copula == "frank"
    return -1/θ*log(1 +(w*(1-exp(-θ)))/(w*(exp(-θ*u1) -1) -exp(-θ*u1)))
  elseif copula == "amh"
    a = 1 -u1
    b = 1 -θ *(1 +2 *a *w)+2*θ^2*a^2 *w
    c = 1 -θ *(2 -4*w +4 *a *w)+θ^2 *(1 -4 *a *w+4 *a^2 *w)
    return 2*w *(a *θ -1)^2 /(b+sqrt(c))
  end
end
"""
    Chain_of_Archimedeans

Subsequent pairs of marginals are modeled by bi-variate copulas form the Archimedean
family, following copulas are supported: "Clayton", "Frank", "Ali-Mikhail-Haq",
"Gumbel" copula is not supported.

Fields:
  - n::Int - number of variables
  - θ::Vector{Real} - a vector of parameters
  - copulas::Vector{String} - a vector indicating bi-variate copulas.
  possible elements "clayton", "frank", "amh"

Constructor

    Chain_of_Archimedeans(θ::Vector{Real}, copulas::Vector{String})

requires length(θ) = length(copulas) and limitations on θ for particular bi-variate copulas

Constructor

    Chain_of_Archimedeans(θ::Vector{Real}, copulas::String)
one copula family for all subsequent pairs of marginals.

Constructors

    Chain_of_Archimedeans(θ::Vector{Real}, copulas::Vector{String}, cor::Type{<:CorrelationType})

    Chain_of_Archimedeans(θ::Vector{Real}, copulas::String, cor::Type{<:CorrelationType})

For computing copula parameters from expected correlation use empty type cor::Type{<:CorrelationType}.
cor ∈ {SpearmanCorrelation, KendallCorrelation} uses these correlations in the place of θ  in the constructor
to compute θ.

In all cases n = length(θ)+1.

```jldoctest

julia> c = Chain_of_Archimedeans([4., 11.], "frank")
Chain_of_Archimedeans(3, [4.0, 11.0], ["frank", "frank"])

julia> c = Chain_of_Archimedeans([.5, .7], ["frank", "clayton"], "Kendall")
Chain_of_Archimedeans(3, [5.736282707019972, 4.666666666666666], ["frank", "clayton"])

```
"""
struct Chain_of_Archimedeans{T}
  n::Int
  θ::Vector{T}
  copulas::Vector{String}
  function(::Type{Chain_of_Archimedeans})(θ::Vector{T}, copulas::Vector{String}) where T <: Real
      n = length(θ)+1
      n-1 == length(copulas) || throw(BoundsError("length(θ) ≠ length(copulas)"))
      map(i -> testbivθ(θ[i], copulas[i]), 1:n-1)
      for copula in copulas
        copula in ["clayton", "amh", "frank"] || throw(AssertionError("$(copula) copula is not supported"))
      end
      new{T}(n, θ, copulas)
  end
  function(::Type{Chain_of_Archimedeans})(θ::Vector{T}, copulas::Vector{String}, cor::Type{<:CorrelationType}) where T <: Real
      n = length(θ)+1
      n - 1 == length(copulas) || throw(BoundsError("length(θ) ≠ length(copulas)"))
      for copula in copulas
        copula in ["clayton", "amh", "frank"] || throw(AssertionError("$(copula) copula is not supported"))
      end
      θ = map(i -> usebivρ(θ[i], copulas[i], cor), 1:n-1)
      new{T}(n, θ, copulas)
  end
  function(::Type{Chain_of_Archimedeans})(θ::Vector{T}, copula::String) where T <: Real
      n = length(θ)+1
      map(i -> testbivθ(θ[i], copula), 1:n-1)
      copula in ["clayton", "amh", "frank"] || throw(AssertionError("$(copula) copula is not supported"))
      new{T}(n, θ, [copula for i in 1:n-1])
  end
  function(::Type{Chain_of_Archimedeans})(θ::Vector{Real}, copula::String, cor::Type{<:CorrelationType}) where T <: Real
      n = length(θ)+1
      map(i -> testbivθ(θ[i], copula), 1:n-1)
      copula in ["clayton", "amh", "frank"] || throw(AssertionError("$(copula) copula is not supported"))
      θ = map(i -> usebivρ(θ[i], copula, cor), 1:n-1)
      new{T}(n, θ, [copula for i in 1:n-1])
  end
end

"""
    simulate_copula(t::Int, copula::Chain_of_Archimedeans; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of multivariate data modeled by the chain of bivariate
Archimedean copulas, i.e.

    Chain_of_Archimedeans(θ::Vector{Flota64}, copulas::Union{String, Vector{String}})

```jldoctest
julia> Random.seed!(43);

julia> c = Chain_of_Archimedeans([4., 11.], "frank")
Chain_of_Archimedeans(3, [4.0, 11.0], ["frank", "frank"])

julia> simulate_copula(1, c)
1×3 Array{Float64,2}:
 0.180975  0.492923  0.679345

julia> c = Chain_of_Archimedeans([.5, .7], ["frank", "clayton"], KendallCorrelation)
Chain_of_Archimedeans(3, [5.736282707019972, 4.666666666666666], ["frank", "clayton"])

julia> Random.seed!(43);

julia> simulate_copula(1, c)
1×3 Array{Float64,2}:
 0.180975  0.408582  0.646887
```
"""
function simulate_copula(t::Int, copula::Chain_of_Archimedeans{T}; rng::AbstractRNG = Random.GLOBAL_RNG) where T <: Real
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    U
end

"""
    simulate_copula!(U::Matrix{Real}, copula::Chain_of_Archimedeans; rng::AbstractRNG = Random.GLOBAL_RNG)

Given the preallocated output U, returns size(U,1) realizations of data modeled by the chain
of bivariate Archimedean copulas. N.o. marginals is size(U,2), requires size(U,2) == copula.n

```jldoctest
julia> c = Chain_of_Archimedeans([4., 11.], "frank")
Chain_of_Archimedeans(3, [4.0, 11.0], ["frank", "frank"])

julia> u = zeros(1,3)
1×3 Array{Float64,2}:
 0.0  0.0  0.0

julia> Random.seed!(43);

julia> simulate_copula!(u,c)

julia> u
1×3 Array{Float64,2}:
 0.180975  0.492923  0.679345

 julia> u = zeros(1,3)
1×3 Array{Float64,2}:
 0.0  0.0  0.0

julia> Random.seed!(43);

julia> c = Chain_of_Archimedeans([.5, .7], ["frank", "clayton"], KendallCorrelation)
Chain_of_Archimedeans(3, [5.736282707019972, 4.666666666666666], ["frank", "clayton"])

julia> simulate_copula!(u,c)

julia> u
1×3 Array{Float64,2}:
 0.180975  0.408582  0.646887
```
"""
function simulate_copula!(U::Matrix{T}, copula::Chain_of_Archimedeans{T}; rng::AbstractRNG = Random.GLOBAL_RNG) where T <: Real
    θ = copula.θ
    copulas = copula.copulas
    size(U, 2) == copula.n || throw(AssertionError("n.o. margins in pre allocated output and copula not equal"))
    for j in 1:size(U,1)
      u = rand(rng, T)
      w = rand(rng, T, copula.n-1)
      for i in 1:copula.n-1
        u = hcat(u, rand2cop(u[i], θ[i], copulas[i], w[i]))
      end
      U[j,:] = u
    end
end

"""
  testbivθ(θ::Union{Real}, copula::String)

Tests bivariate copula parameter

clayton bivariate sub-copulas with parameters (θᵢ ≥ -1) ^ ∧ (θᵢ ≠ 0).
amh -- Ali-Mikhail-Haq bi-variate sub-copulas with parameters -1 ≥ θᵢ ≥ 1
Frank bi-variate sub-copulas with parameters (θᵢ ≠ 0)
"""
function testbivθ(θ::Union{T, Int}, copula::String) where T <: Real
  !(0. in θ)|(copula == "amh") || throw(DomainError("not supported for θ = 0"))
  if copula == "clayton"
    θ >= -1 || throw(DomainError("not supported for θ < -1"))
  elseif copula == "amh"
    -1 <= θ <= 1|| throw(DomainError("amh biv. copula supported only for -1 ≤ θ ≤ 1"))
  end
  Nothing
end

"""
    usebivρ(ρ::Real, copula::String, cor::Type{<:CorrelationType})

Returns Real, a copula parameter given the Spearman or the Kendall correlation

For Clayton or Frank copula the correlation must fulfill (-1 > ρᵢ > 1) ∧ (ρᵢ ≠ 0)
For the AMH copula Spearman must fulfill -0.2816 > ρᵢ >= .5, while Kendall -0.18 < τ < 1/3
"""
function usebivρ(ρ::T, copula::String, cor::Type{<:CorrelationType}) where T <: Real
  if copula == "amh"
      -0.2816 < ρ <= 0.5 || throw(DomainError("correlation coeficiant must fulfill -0.2816 < ρ <= 0.5"))
    if cor == KendallCorrelation
      -0.18 < ρ < 1/3 || throw(DomainError("correlation coeficiant must fulfill -0.2816 < ρ <= 1/3"))
    end
  else
    -1 < ρ < 1 || throw(DomainError("correlation coeficiant must fulfill -1 < ρ < 1"))
    !(0. in ρ) || throw(DomainError("not supported for ρ = 0"))
  end
  if cor == KendallCorrelation
      return τ2θ(ρ, copula)
  elseif cor == SpearmanCorrelation
      return ρ2θ(ρ, copula)
  end
end


# chain frechet copulas
"""
    Chain_of_Frechet

Chain of bi-variate Frechet copulas. Models each subsequent pair of marginals by the
bi-variate Frechet copula.

Fields:
    - n::Int - number of marginals
    - α::Vector{Real}  - vector of parameters for the maximal copula
    - β::Vector{Real} - vector of parameters for the minimal copula

Here α[i] and β[i] parameterized bi-variate Frechet copula between i th and i+1 th marginals.

Constructors

    Chain_of_Frechet(α::Vector{Real})
here β = zero(0)

    Chain_of_Frechet(α::Vector{Real}, β::Vector{Real})

```jldoctest
julia> Chain_of_Frechet([0.2, 0.3, 0.4])
Chain_of_Frechet(4, [0.2, 0.3, 0.4], [0.0, 0.0, 0.0])

julia> Chain_of_Frechet([0.2, 0.3, 0.4], [0.1, 0.1, 0.1])
Chain_of_Frechet(4, [0.2, 0.3, 0.4], [0.1, 0.1, 0.1])
```
"""
struct Chain_of_Frechet{T}
  n::Int
  α::Vector{T}
  β::Vector{T}
  function(::Type{Chain_of_Frechet})(α::Vector{T}) where T <: Real
      n = length(α)+1
      β = zero(α)
      minimum(α) >= 0 || throw(DomainError("negative α parameter"))
      maximum(α) <= 1 || throw(DomainError("α parameter greater than 1"))
      new{T}(n, α, β)
  end
  function(::Type{Chain_of_Frechet})(α::Vector{T}, β::Vector{T}) where T <: Real
      n = length(α)+1
      n == length(β) +1 || throw(AssertionError("length(α) ≠ length(β)"))
      minimum(α) >= 0 || throw(DomainError("negative α parameter"))
      minimum(β) >= 0 || throw(DomainError("negative β parameter"))
      maximum(α+β) <= 1 || throw(DomainError("α[i] + β[i] > 0"))
      new{T}(n, α, β)
  end
end

"""
    simulate_copula(t::Int, copula::Chain_of_Frechet; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations modeled by the chain of bivariate two parameter Frechet copulas

```jldoctest
julia> Random.seed!(43)

julia> simulate_copula(10, Chain_of_Frechet([0.6, 0.4], [0.3, 0.5]))
  10×3 Array{Float64,2}:
  0.996764  0.996764  0.996764
  0.204033  0.795967  0.204033
  0.979901  0.979901  0.0200985
  0.120669  0.879331  0.120669
  0.453027  0.453027  0.453027
  0.800909  0.199091  0.800909
  0.54892   0.54892   0.54892
  0.933832  0.933832  0.0661679
  0.396943  0.396943  0.396943
  0.804096  0.851275  0.955881
```
"""
function simulate_copula(t::Int, copula::Chain_of_Frechet{T}; rng::AbstractRNG = Random.GLOBAL_RNG) where T <: Real
  α = copula.α
  β = copula.β
  n = copula.n
  fncopulagen(α, β, rand(rng, t, n))
end

"""
  fncopulagen(α::Vector{Real}, β::Vector{Real}, u::Matrix{Real})

```jldoctest

julia> fncopulagen(2, [0.2, 0.4], [0.1, 0.1], [0.2 0.4 0.6; 0.3 0.5 0.7])
2×3 Array{Float64,2}:
 0.6  0.4  0.2
 0.7  0.5  0.3

```
"""
function fncopulagen(α::Vector{T}, β::Vector{T}, u::Matrix{T}) where T <: Real
  p = invperm(sortperm(u[:,1]))
  u = u[:,end:-1:1]
  lx = floor.(Int, size(u,1).*α)
  li = floor.(Int, size(u,1).*β) + lx
  for j in 1:size(u, 2)-1
    u[p[1:lx[j]],j+1] = u[p[1:lx[j]], j]
    r = p[lx[j]+1:li[j]]
    u[r,j+1] = 1 .-u[r,j]
  end
  u
end
