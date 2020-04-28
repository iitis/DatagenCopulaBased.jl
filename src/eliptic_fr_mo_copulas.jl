# Elliptical Frechet and Marshal Olkim copulas generators

## Elliptical copulas

"""
    Gaussian_cop

Gaussian copula

Fields:
   - Σ - the correlation matrix must be symmetric, positively defined and with ones on diagonal

Constructor

    Gaussian_cop(Σ::Matrix{Float64})

```jldoctest

julia> Gaussian_cop([1. 0.5; 0.5 1.])
Gaussian_cop([1.0 0.5; 0.5 1.0])

```
"""
struct Gaussian_cop
  Σ::Matrix{Float64}
  function(::Type{Gaussian_cop})(Σ::Matrix{Float64})
    Σ ≈ transpose(Σ) || throw(DomainError("Σ matrix not symmetric"))
    isposdef(Σ) || throw(DomainError("Σ matrix not positivelly defined"))
    prod(diag(Σ)) ≈ 1.0 || throw(DomainError("Σ matrix do not have ones on diagonal"))
    new(Σ)
  end
end

"""
    simulate_copula(t::Int, copula::Gaussian_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of the Gaussian copula

    Gaussian_cop(Σ)

```jldoctest

julia> Random.seed!(43);

julia> simulate_copula(10, Gaussian_cop([1. 0.5; 0.5 1.]))
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
function simulate_copula(t::Int, copula::Gaussian_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  Σ = copula.Σ
  z = transpose(rand(rng, MvNormal(Σ),t))
  for i in 1:size(Σ, 1)
    d = Normal(0, sqrt.(Σ[i,i]))
    z[:,i] = cdf.(d, z[:,i])
  end
  Array(z)
end

"""
    Student_cop

t-Student copula

fields
  - Σ::Matrix{Float64} - the correlation matrix must be symmetric, positively defined and with ones on diagonal
  - ν::Int - the parameter n.o. degrees of freedom we require ν > 0

Constructor:

    Student_cop(Σ::Matrix{Float64}, ν::Int)

```jldoctest

julia> Student_cop([1. 0.5; 0.5 1.], 4)
Student_cop([1.0 0.5; 0.5 1.0], 4)

```
"""
struct Student_cop
  Σ::Matrix{Float64}
  ν::Int
  function(::Type{Student_cop})(Σ::Matrix{Float64}, ν::Int)
    Σ ≈ transpose(Σ) || throw(DomainError("Σ matrix not symmetric"))
    isposdef(Σ) || throw(DomainError("Σ matrix not positivelly defined"))
    prod(diag(Σ)) ≈ 1.0 || throw(DomainError("Σ matrix do not have ones on diagonal"))
    ν > 0 || throw(DomainError("ν lower or equal zero"))
    new(Σ, ν)
  end
end

"""
    simulate_copula(t::Int, copula::Student _cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of the t-Student Copula

    Student_cop(Σ, ν)

where: Σ - correlation matrix, ν - degrees of freedom.

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(10, Student_cop([1. 0.5; 0.5 1.], 10))
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
function simulate_copula(t::Int, copula::Student_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  Σ = copula.Σ
  ν = copula.ν
  z = transpose(rand(rng, MvNormal(Σ),t))
  U = rand(rng, Chisq(ν), size(z, 1))
  for i in 1:size(Σ, 1)
    x = z[:,i].*sqrt.(ν./U)./sqrt(Σ[i,i])
    z[:,i] = cdf.(TDist(ν), x)
  end
  Array(z)
end

"""
    Frechet_cop

The Frechet copula

Fileds:
  - n - number of marginals
  - α - the parameter of the maximal copula
  - β - the parameter of the minimal copula

Constructor

    Frechet_cop(n::Int, α::Float64)

The one parameter Frechet copula is a combination of maximal copula with  weight α
and independent copula with  weight 1-α.

Constructor

    Frechet_cop(n::Int, α::Float64, β::Float64)

The two parameters Frechet copula C = α C{max} + β C{min} + (1- α - β) C{⟂}, supported
only for n = 2.

```jldoctest
julia> Frechet_cop(4, 0.5)
Frechet_cop(4, 0.5, 0.0)

julia> Frechet_cop(2, 0.5, 0.3)
Frechet_cop(2, 0.5, 0.3)
```
"""
struct Frechet_cop
  n::Int
  α::Float64
  β::Float64
  function(::Type{Frechet_cop})(n::Int, α::Float64)
    0 <= α <= 1 || throw(DomainError("generaton not supported for α ∉ [0,1]"))
    n > 1 || throw(DomainError("n must be greater than 1"))
    new(n, α, 0.)
  end
  function(::Type{Frechet_cop})(n::Int, α::Float64, β::Float64)
    0 <= α <= 1 || throw(DomainError("generaton not supported for α ∉ [0,1]"))
    0 <= β <= 1 || throw(DomainError("generaton not supported for β ∉ [0,1]"))
    n == 2 || throw(AssertionError("two parameters Frechet copula supported only for n = 2"))
    0 <= α+β <= 1 || throw(DomainError("α+β must be in range [0,1]"))
    new(n, α, β)
  end
end

"""
    simulate_copula(t::Int, copula::Frechet_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizatioins of data from the Frechet copula

    Frechet(n, α)
    Frechet(n, α, β)

```jldoctest

julia> Random.seed!(43);

julia> f = Frechet_cop(3, 0.5)
Frechet_cop(3, 0.5, 0.0)

julia> simulate_copula(1, f)
1×3 Array{Float64,2}:
0.180975  0.775377  0.888934

julia> Random.seed!(43);

julia> f = Frechet_cop(2, 0.5, 0.2)
Frechet_cop(2, 0.5, 0.2)

julia> simulate_copula(1, f)
1×2 Array{Float64,2}:
0.180975  0.775377
```
"""

function simulate_copula(t::Int, copula::Frechet_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  n = copula.n
  α = copula.α
  β = copula.β
  u = zeros(t,n)
  if (β > 0) & (n == 2)
    for j in 1:t
      u_el = rand(rng, n)
      frechet_el2!(u_el, α, β, rand(rng))
      u[j,:] = u_el
    end
  else
    for j in 1:t
      u_el = rand(rng, n)
      frechet_el!(u_el, α, rand(rng))
      u[j,:] = u_el
    end
  end
  return u
end

"""
  frechet(t::Int, n::Int, α::Float64)

Given n-variate random data u ∈ R^{t, n}
Returns t realization of n variate data generated from one parameter Frechet_cop(n, α).

```jldoctest
julia> Random.seed!(43);

julia> u = rand(10, 2);

julia> frechet(0.5, u)
10×2 Array{Float64,2}:
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
function frechet(α::Float64, u::Matrix{Float64}; rng::AbstractRNG = Random.GLOBAL_RNG)
  for j in 1:size(u, 1)
    v = rand(rng)
    el = u[j, :]
    frechet_el!(el, α, v)
    u[j,:] = el
  end
  u
end

"""
  frechet_el!(u::Vector{Float64}, α::Float64, v::Float64 = rand())

Given n-variate random vector changes it to such modeled by the two parameters Frechet_cop(n, α, β).

"""
function frechet_el!(u::Vector{Float64}, α::Float64, v::Float64 = rand())
  if (α >= v)
    for i in 1:length(u)-1
      u[i] = u[end]
    end
  end
end

"""
  function frechet_el2!(u::Vector{Float64}, α::Float64, β::Float64, v::Float64 = rand())

Given bivariate random vector changes it to such modeled by the two parameters Frechet_cop(n, α, β).

"""
function frechet_el2!(u::Vector{Float64}, α::Float64, β::Float64, v::Float64 = rand())
  if (α >= v)
    u[1] = u[2]
  elseif (α < v <= α+β)
    u[1] = 1-u[2]
  end
end

### Marshall - Olkin familly

"""
    Marshall_Olkin_cop

Fields:
  - n::Int - number of marginals
  - λ::Vector{Float64} - vector of non-negative parameters λₛ, i.e.:
      λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]
      and n = ceil(Int, log(2, length(λ)-1)).

Constructor

    Marshall_Olkin_cop(λ)

length(λ) ≧ 3 is required

```jldoctest
julia> Marshall_Olkin_cop([0.5, 0.5, 0.6])
Marshall_Olkin_cop(2, [0.5, 0.5, 0.6])

julia> Marshall_Olkin_cop([0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8])
Marshall_Olkin_cop(3, [0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8])
```
"""
struct Marshall_Olkin_cop
  n::Int
  λ::Vector{Float64}
  function(::Type{Marshall_Olkin_cop})(λ::Vector{Float64})
    minimum(λ) >= 0 || throw(AssertionError("all parameters must by >= 0 "))
    length(λ) >= 3 || throw(AssertionError("not supported for length(λ) < 3"))
    n = floor(Int, log(2, length(λ)+1))
    new(n, λ)
  end
end

"""
    simulate_copula(t::Int, copula::Marshall_Olkin_cop(λ); rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of the n-variate Marshall-Olkin copula:

    Marshall_Olkin_cop(λ)

```julia> Random.seed!(43);

julia> f = Marshall_Olkin_cop([1.,2.,3.])
Marshall_Olkin_cop(2, [1.0, 2.0, 3.0])

julia> simulate_copula(1, f)
1×2 Array{Float64,2}:
  0.854724  0.821831
```
"""
function simulate_copula(t::Int, copula::Marshall_Olkin_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  λ = copula.λ
  n = copula.n
  U = zeros(t, n)
  s = collect(combinations(1:n))
  for j in 1:t
    u = rand(rng, 2^n-1)
    U[j,:] = mocopula_el(u, n, λ, s)
  end
  U
end

"""
  mocopula(u::Matrix{Float64}, n::Int, λ::Vector{Float64})

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
function mocopula(u::Matrix{Float64}, n::Int, λ::Vector{Float64})
  t = size(u,1)
  U = zeros(t, n)
  s = collect(combinations(1:n))
  for j in 1:t
      U[j,:] = mocopula_el(u[j,:], n, λ, s)
  end
  U
end

"""
    mocopula_el(u::Vector{Float64}, n::Int, λ::Vector{Float64}, s::Vector{Vector{Int}})

```jldoctest

julia> mocopula_el([0.1, 0.2, 0.3], 2, [1., 2., 3.], s)
2-element Array{Float64,1}:
 0.20082988502465082
 0.1344421423967149
```
"""
function mocopula_el(u::Vector{Float64}, n::Int, λ::Vector{Float64}, s::Vector{Vector{Int}})
  l = length(u)
  U = zeros(n)
  for i in 1:n
    inds = findall([i in s[k] for k in 1:l])
    x = minimum([-log(u[k])./(λ[k]) for k in inds])
    Λ = sum(λ[inds])
    U[i] = exp.(-Λ*x)
  end
    U
end
