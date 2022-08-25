# Elliptical Frechet and Marshal Olkim copulas generators

## Elliptical copulas

"""
    GaussianCopula

Gaussian copula

Fields:
   - Σ - the correlation matrix must be symmetric, positively defined and with ones on diagonal

Constructor

    GaussianCopula(Σ::Matrix{T}) where T <: Real

```jldoctest

julia> GaussianCopula([1. 0.5; 0.5 1.])
GaussianCopula([1.0 0.5; 0.5 1.0])

```
"""
struct GaussianCopula{T} <: Copula{T}
  Σ::Matrix{T}
  n::Int
  function(::Type{GaussianCopula})(Σ::Matrix{T}) where T <: Real
    Σ ≈ transpose(Σ) || throw(DomainError("Σ matrix not symmetric"))
    isposdef(Σ) || throw(DomainError("Σ matrix not positivelly defined"))
    prod(diag(Σ)) ≈ 1.0 || throw(DomainError("Σ matrix do not have ones on diagonal"))
    new{T}(Σ,size(Σ,1))
  end
end


function simulate_copula!(U, copula::GaussianCopula{T}; rng = Random.GLOBAL_RNG) where T
  Σ = copula.Σ
  z = transpose(rand(rng, MvNormal(Σ),size(U,1)))
  for i in 1:size(Σ, 1)
    d = Normal(0, sqrt.(Σ[i,i]))
    z[:,i] = cdf.(d, z[:,i])
  end
  U .= Array(z)
  return nothing
end

"""
    StudentCopula

t-Student copula

fields
  - Σ::Matrix{Real} - the correlation matrix must be symmetric, positively defined and with ones on diagonal
  - ν::Int - the parameter n.o. degrees of freedom we require ν > 0

Constructor:

    StudentCopula(Σ::Matrix{Real}, ν::Int)

```jldoctest

julia> StudentCopula([1. 0.5; 0.5 1.], 4)
StudentCopula([1.0 0.5; 0.5 1.0], 4)

```
"""
struct StudentCopula{T} <: Copula{T}
  Σ::Matrix{T}
  ν::Int
  n::Int
  function(::Type{StudentCopula})(Σ::Matrix{T}, ν::Int) where T <: Real
    Σ ≈ transpose(Σ) || throw(DomainError("Σ matrix not symmetric"))
    isposdef(Σ) || throw(DomainError("Σ matrix not positivelly defined"))
    prod(diag(Σ)) ≈ 1.0 || throw(DomainError("Σ matrix do not have ones on diagonal"))
    ν > 0 || throw(DomainError("ν lower or equal zero"))
    new{T}(Σ, ν,size(Σ,1))
  end
end


function simulate_copula!(U, copula::StudentCopula{T}; rng = Random.GLOBAL_RNG) where T
  Σ = copula.Σ
  ν = copula.ν
  z = transpose(rand(rng, MvNormal(Σ),size(U,1)))
  V = rand(rng, Chisq(ν), size(z, 1))
  V = T.(V)
  for i in 1:size(Σ, 1)
    x = z[:,i].*sqrt.(ν./V)./sqrt(Σ[i,i])
    z[:,i] = cdf.(TDist(ν), x)
  end
  U .= Array(z)
  return nothing
end

"""
    FrechetCopula

The Frechet copula

Fileds:
  - n - number of marginals
  - α - the parameter of the maximal copula
  - β - the parameter of the minimal copula

Constructor

    FrechetCopula(n::Int, α::Real)

The one parameter Frechet copula is a combination of maximal copula with  weight α
and independent copula with  weight 1-α.

Constructor

    FrechetCopula(n::Int, α::Real, β::Real)

The two parameters Frechet copula C = α C{max} + β C{min} + (1- α - β) C{⟂}, supported
only for n = 2.

```jldoctest
julia> FrechetCopula(4, 0.5)
FrechetCopula(4, 0.5, 0.0)

julia> FrechetCopula(2, 0.5, 0.3)
FrechetCopula(2, 0.5, 0.3)
```
"""
struct FrechetCopula{T} <: Copula{T}
  n::Int
  α::T
  β::T
  function(::Type{FrechetCopula})(n::Int, α::T) where T <: Real
    0 <= α <= 1 || throw(DomainError("generaton not supported for α ∉ [0,1]"))
    n > 1 || throw(DomainError("n must be greater than 1"))
    new{T}(n, α, 0.)
  end
  function(::Type{FrechetCopula})(n::Int, α::T, β::T) where T <: Real
    0 <= α <= 1 || throw(DomainError("generaton not supported for α ∉ [0,1]"))
    0 <= β <= 1 || throw(DomainError("generaton not supported for β ∉ [0,1]"))
    n == 2 || throw(AssertionError("two parameters Frechet copula supported only for n = 2"))
    0 <= α+β <= 1 || throw(DomainError("α+β must be in range [0,1]"))
    new{T}(n, α, β)
  end
end



"""
    simulate_copula!(U::Matrix{Real}, copula::FrechetCopula; rng::AbstractRNG = Random.GLOBAL_RNG)

Given the preallocated output U, Returns size(U,1) realizations from the Frechet copula - FrechetCopula
N.o. marginals is size(U,2), requires size(U,2) == copula.n

```jldoctest
julia> f = FrechetCopula(3, 0.5)
FrechetCopula(3, 0.5, 0.0)

julia> u = zeros(1,3)
1×3 Array{Real,2}:
 0.0  0.0  0.0

julia> Random.seed!(43);

julia> simulate_copula!(u,f)

julia> u
1×3 Array{Real,2}:
 0.180975  0.775377  0.888934
```
"""
function simulate_copula!(U, copula::FrechetCopula{T}; rng = Random.GLOBAL_RNG) where T
  n = copula.n
  α = copula.α
  β = copula.β
  size(U, 2) == n || throw(AssertionError("n.o. margins in pre allocated output and copula not equal"))
  if (β > 0) & (n == 2)
    for j in 1:size(U,1)
      u_el = rand(rng, T, n)
      frechet_el2!(u_el, α, β, rand(rng, T))
      U[j,:] = u_el
    end
  else
    for j in 1:size(U,1)
      u_el = rand(rng, T, n)
      frechet_el!(u_el, α, rand(rng))
      U[j,:] = u_el
    end
  end
end

"""
  frechet(t::Int, n::Int, α::Real; rng::AbstractRNG)

Given n-variate random data u ∈ R^{t, n}
Returns t realization of n variate data generated from one parameter FrechetCopula(n, α).

```jldoctest
julia> Random.seed!(43);

julia> u = rand(10, 2);

julia> frechet(0.5, u)
10×2 Array{Real,2}:
 0.180975   0.661781
 0.0742681  0.0742681
 0.888934   0.125437
 0.0950087  0.0950087
 0.130474   0.130474
 0.912603   0.740184
 0.828727   0.00463791
 0.400537   0.0288987
 0.521601   0.521601
 0.955881   0.851275
```
"""
function frechet(α::T, u; rng) where T
  for j in 1:size(u, 1)
    v = rand(rng, T)
    el = u[j, :]
    frechet_el!(el, α, v)
    u[j,:] = el
  end
  u
end

"""
  frechet_el!(u::Vector{Real}, α::Real, v::Real)

Given n-variate random vector changes it to such modeled by the two parameters FrechetCopula(n, α, β).
v is the random number form [0,1].
"""
function frechet_el!(u, α, v)
  if (α >= v)
    for i in 1:length(u)-1
      u[i] = u[end]
    end
  end
end

"""
  function frechet_el2!(u::Vector{Real}, α::Real, β::Real, v::Real)

Given bivariate random vector changes it to such modeled by the two parameters FrechetCopula(n, α, β).
v is the random number form [0,1]

"""
function frechet_el2!(u, α, β, v)
  if (α >= v)
    u[1] = u[2]
  elseif (α < v <= α+β)
    u[1] = 1-u[2]
  end
end

### Marshall - Olkin familly

"""
    MarshallOlkinCopula

Fields:
  - n::Int - number of marginals
  - λ::Vector{Real} - vector of non-negative parameters λₛ, i.e.:
      λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]
      and n = ceil(Int, log(2, length(λ)-1)).

Constructor

    MarshallOlkinCopula(λ)

length(λ) ≧ 3 is required

```jldoctest
julia> MarshallOlkinCopula([0.5, 0.5, 0.6])
MarshallOlkinCopula(2, [0.5, 0.5, 0.6])

julia> MarshallOlkinCopula([0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8])
MarshallOlkinCopula(3, [0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8])
```
"""
struct MarshallOlkinCopula{T} <: Copula{T}
  n::Int
  λ::Vector{T}
  function(::Type{MarshallOlkinCopula})(λ::Vector{T}) where T <: Real
    minimum(λ) >= 0 || throw(AssertionError("all parameters must by >= 0 "))
    length(λ) >= 3 || throw(AssertionError("not supported for length(λ) < 3"))
    n = floor(Int, log(2, length(λ)+1))
    new{T}(n, λ)
  end
end



"""
    simulate_copula!(U::Matrix{Real}, copula::MarshallOlkinCopula; rng::AbstractRNG = Random.GLOBAL_RNG)

Given the preallocated output U, Returns size(U,1) realizations from the Marshall  Olkin copula - MarshallOlkinCopula
N.o. marginals is size(U,2), requires size(U,2) == copula.n

```jldoctest
julia> u = zeros(1,2)
1×2 Array{Float64,2}:
 0.0  0.0

julia> cop = MarshallOlkinCopula([1.,2.,3.])
MarshallOlkinCopula(2, [1.0, 2.0, 3.0])

julia> Random.seed!(43);

julia> simulate_copula!(u,cop)

julia> u
1×2 Array{Float64,2}:
 0.854724  0.821831
```
"""
function simulate_copula!(U, copula::MarshallOlkinCopula{T}; rng = Random.GLOBAL_RNG) where T
  λ = copula.λ
  n = copula.n
  size(U, 2) == n || throw(AssertionError("n.o. margins in pre allocated output and copula not equal"))
  s = collect(combinations(1:n))
  for j in 1:size(U,1)
    u = rand(rng, T, 2^n-1)
    U[j,:] = mocopula_el(u, n, λ, s)
  end
end

"""
  mocopula(u::Matrix{Real}, n::Int, λ::Vector{Real})

  Returns: t x n Matrix{Float}, t realizations of n-variate data generated from Marshall-Olkin
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
function mocopula(u, n, λ)
  T = eltype(u)
  t = size(u,1)
  U = zeros(T, t, n)
  s = collect(combinations(1:n))
  for j in 1:t
      U[j,:] = mocopula_el(u[j,:], n, λ, s)
  end
  U
end

"""
    mocopula_el(u::Vector{Real}, n::Int, λ::Vector{Real}, s::Vector{Vector{Int}})

```jldoctest

julia> mocopula_el([0.1, 0.2, 0.3], 2, [1., 2., 3.], s)
2-element Array{Float64,1}:
 0.20082988502465082
 0.1344421423967149
```
"""
function mocopula_el(u, n, λ, s)
  T = eltype(u)
  l = length(u)
  U = zeros(T, n)
  for i in 1:n
    inds = findall([i in s[k] for k in 1:l])
    x = minimum([-log(u[k])./(λ[k]) for k in inds])
    Λ = sum(λ[inds])
    U[i] = exp.(-Λ*x)
  end
    U
end
