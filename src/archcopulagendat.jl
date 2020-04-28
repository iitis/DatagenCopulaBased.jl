# Archimedean copulas


#=
"""
  getV0(θ::Float64, v::Vector{Float64}, copula::String)

Returns Vector{Float} or Vector{Int} of realizations of axiliary variable V0
used to ganarate data from 1d archimedean copula with parameter Θ, given v:
realizations of 1d variable uniformly distributed on [0,1]

```jldoctest

julia> getV0(2., [0.2, 0.4, 0.6, 0.8], "clayton")
4-element Array{Float64,1}:
 0.0641848
 0.274996
 0.708326
 1.64237
```
"""
function getV0(θ::Float64, v::Union{Float64, Vector{Float64}, Matrix{Float64}}, copula::String)
  if copula == "clayton"
    return quantile.(Gamma(1/θ, 1), v)
  elseif copula == "amh"
    return 1 .+quantile.(Geometric(1-θ), v)
  elseif copula == "frank"
    return logseriesquantile(1-exp(-θ), v)
  elseif copula == "gumbel"
    return levyel(θ, v[1], v[2])
  end
  throw(AssertionError("$(copula) not supported"))
end

"""
  phi(u::Matrix{Float64}, θ::Float64, copula::String)

Given a matrix t realizations of n variate data R^{t x n} ∋ u = -log(rand(t,n))./V0
returns it transformed through an inverse generator of archimedean copula. Output
is distributed uniformly on [0,1]ⁿ.

```jldoctest

julia> phi([0.2 0.6; 0.4 0.8], 2., "clayton")
2×2 Array{Float64,2}:
 0.845154  0.6742
 0.745356  0.620174
```
"""
function phi(u::Union{Vector{Float64}, Matrix{Float64}}, θ::Float64, copula::String)
  if copula == "clayton"
    return (1 .+ u).^(-1/θ)
  elseif copula == "amh"
    return (1-θ) ./(exp.(u) .-θ)
  elseif copula == "frank"
    return -log.(1 .+exp.(-u) .*(exp(-θ)-1)) ./θ
  elseif copula == "gumbel"
    return exp.(-u.^(1/θ))
  end
  throw(AssertionError("$(copula) not supported"))
end
=#

"""
  arch_gen(copula::String, r::Matrix{Float}, θ::Float64)

Auxiliary function used to generate data from archimedean copula (clayton, gumbel, frank or amh)
parametrized by a single parameter θ given a matrix of independent [0,1] distributerd
random vectors.

```jldoctest
  julia> arch_gen("clayton", [0.2 0.6 0.9; 0.4 0.5 0.8], 2.)
  2×2 Array{Float64,2}:
   0.675778  0.851993
   0.687482  0.736394
```
"""

function arch_gen(copula::String, r::Matrix{Float64}, θ::Float64)
  rng = Random.GLOBAL_RNG
  t = size(r,1)
  if copula == "clayton"
    U = zeros(t, size(r,2)-1)
    for j in 1:t
      U[j,:] = clayton_gen(r[j,:], θ)
    end
    return U
  elseif copula == "amh"
    U = zeros(t, size(r,2)-1)
    for j in 1:t
      U[j,:] = amh_gen(r[j,:], θ)
    end
    return U
  elseif copula == "frank"
    U = zeros(t, size(r,2)-1)
    w = logseriescdf(1-exp(-θ))
    for j in 1:t
      U[j,:] = frank_gen(r[j,:], θ, w)
    end
    return U
  else
    U = zeros(t, size(r,2)-1)
    u = r[:,end]
    p = invperm(sortperm(u))
    v = [levyel(θ, rand(rng), rand(rng)) for i in 1:length(u)]
    v = sort(v)[p]
    for j in 1:t
      U[j,:] = -log.(r[j,1:end-1])./v[j]
    end
    return exp.(-U.^(1/θ))
  end
end

"""
    gumbel_gen(r::Vector{Float64}, θ::Float64)
"""
function gumbel_gen(r::Vector{Float64}, θ::Float64)
    u = -log.(r[1:end-2])./levyel(θ, r[end-1], r[end])
    return exp.(-u.^(1/θ))
end

"""
    clayton_gen(r::Vector{Float64}, θ::Float64)
"""
function clayton_gen(r::Vector{Float64}, θ::Float64)
    u = -log.(r[1:end-1])./quantile.(Gamma(1/θ, 1), r[end])
    return (1 .+ u).^(-1/θ)
end

"""
    amh_gen(r::Vector{Float64}, θ::Float64)
"""
function amh_gen(r::Vector{Float64}, θ::Float64)
    u = -log.(r[1:end-1])./(1 .+quantile.(Geometric(1-θ), r[end]))
    return (1-θ) ./(exp.(u) .-θ)
end

"""
    frank_gen(r::Vector{Float64}, θ::Float64, logseries::Vector{Float64})
"""
function frank_gen(r::Vector{Float64}, θ::Float64, logseries::Vector{Float64})
    logseriesquantile = findlast(logseries .< r[end])
    u = -log.(r[1:end-1])./logseriesquantile
    return -log.(1 .+exp.(-u) .*(exp(-θ)-1)) ./θ
end

"""
    Gumbel_cop

Fields:
  - n::Int - number of marginals
  - θ::Float64 - parameter

Constructor

        Gumbel_cop(n::Int, θ::Float64)

The Gumbel n variate copula is parameterized by θ::Float64 ∈ [1, ∞), supported for n::Int ≧ 2.

Constructor

    Gumbel_cop(n::Int, θ::Float64, cor::String)

where cor == "Spearman", "Kendall" uses these correlations to compute θ, correlations must be greater than zero.

```jldoctest

julia> Gumbel_cop(4, 3.)
Gumbel_cop(4, 3.0)

julia> Gumbel_cop(4, .75, "Kendall")
Gumbel_cop(4, 4.0)
```
"""
struct Gumbel_cop
  n::Int
  θ::Float64
  function(::Type{Gumbel_cop})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      testθ(θ, "gumbel")
      new(n, θ)
  end
  function(::Type{Gumbel_cop})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "gumbel", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the Gumbel copula -  Gumbel_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(2, Gumbel_cop(3, 1.5))
2×3 Array{Float64,2}:
 0.740038  0.918928  0.950674
 0.637826  0.483514  0.123949
```
"""
function simulate_copula(t::Int, copula::Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
    θ = copula.θ
    n = copula.n
    U = zeros(t, n)
    for j in 1:t
      u = rand(rng, n+2)
      U[j,:] = gumbel_gen(u, θ)
    end
    U
end

"""
    Gumbel_cop_rev

Fields:
  - n::Int - number of marginals
  - θ::Float64 - parameter

Constructor

    Gumbel_cop_rev(n::Int, θ::Float64)

The reversed Gumbel copula (reversed means u → 1 .- u),
parameterized by θ::Float64 ∈ [1, ∞), supported for n::Int ≧ 2.

Constructor

    Gumbel_cop_rev(n::Int, θ::Float64, cor::String)

where cor == "Spearman", "Kendall", uses these correlations to compute θ,
correlations must be greater than zero.

TODO correct
```jldoctest
julia> Gumbel_cop_rev(4, .75, "Kendall")
Gumbel_cop_rev(4, 4.0)

julia> Gumbel_cop_rev(4, 3.)
Gumbel_cop_rev(4, 3.0)

```
"""
struct Gumbel_cop_rev
  n::Int
  θ::Float64
  function(::Type{Gumbel_cop_rev})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      testθ(θ, "gumbel")
      new(n, θ)
  end
  function(::Type{Gumbel_cop_rev})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "gumbel", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::Gumbel_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the Gumbel _cop _rev(n, θ) - the reversed Gumbel copula (reversed means u → 1 .- u).

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(2, Gumbel_cop_rev(3, 1.5))
2×3 Array{Float64,2}:
 0.259962  0.081072  0.0493259
 0.362174  0.516486  0.876051
```
"""
function simulate_copula(t::Int, copula::Gumbel_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)
    θ = copula.θ
    n = copula.n
    return 1 .- simulate_copula(t, Gumbel_cop(n, θ); rng = rng)
end

"""
    Clayton_cop

Fields:
  - n::Int - number of marginals
  - θ::Float64 - parameter

Constructor

    Clayton_cop(n::Int, θ::Float64)

The Clayton n variate copula parameterized by θ::Float64, such that θ ∈ (0, ∞) for n > 2 and θ ∈ [-1, 0) ∪ (0, ∞) for n = 2,
supported for n::Int ≥ 2.

Constructor

    Clayton_cop(n::Int, θ::Float64, cor::String)

uses cor == "Spearman", "Kendall" to compute θ, correlations must be greater than zero.

```jldoctest
julia> Clayton_cop(4, 3.)
Clayton_cop(4, 3.0)

julia> Clayton_cop(4, 0.9, "Spearman")
Clayton_cop(4, 5.5595567742323775)
```
"""
struct Clayton_cop
  n::Int
  θ::Float64
  function(::Type{Clayton_cop})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      if n > 2
        testθ(θ, "clayton")
      else
        (θ >= -1.) & (θ != 0.) || throw(DomainError("bivariate Clayton not supported for θ < -1 or θ = 0"))
      end
      new(n, θ)
  end
  function(::Type{Clayton_cop})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "clayton", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::Clayton_cop; rng::AbstractRNG = Random.GLOBAL_RNG)


Returns t realizations from the Clayton copula - Clayton_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(10, Clayton_cop(2, 1))
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

 julia> Random.seed!(43);

 julia> simulate_copula(2, Clayton_cop(2, -0.5))
 2×2 Array{Float64,2}:
  0.180975  0.907735
  0.775377  0.872074
```
"""
function simulate_copula(t::Int, copula::Clayton_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
    θ = copula.θ
    n = copula.n
  if (n == 2) & (θ < 0)
    return simulate_copula(t, Chain_of_Archimedeans([θ], "clayton"); rng = rng)
  else
    U = zeros(t, n)
    for j in 1:t
      u = rand(rng, n+1)
      U[j,:] = clayton_gen(u, θ)
    end
    return U
  end
end

"""
    Clayton_cop_rev

Fields:
- n::Int - number of marginals
- θ::Float64 - parameter

Constructor

    Clayton_cop_rev(n::Int, θ::Float64)

The reversed Clayton copula parameterized by θ::Float64 (reversed means u → 1 .- u).
Domain: θ ∈ (0, ∞) for n > 2 and θ ∈ [-1, 0) ∪ (0, ∞) for n = 2,
supported for n::Int ≧ 2.

Constructor

    Clayton_cop_rev(n::Int, θ::Float64, cor::String)

uses cor == "Spearman", "Kendall"  to compute θ, correlations must be greater than zero.

```jldoctest

julia> Clayton_cop_rev(4, 3.)
Clayton_cop_rev(4, 3.0)

julia> Clayton_cop_rev(4, 0.9, "Spearman")
Clayton_cop_rev(4, 5.5595567742323775)

```
"""
struct Clayton_cop_rev
  n::Int
  θ::Float64
  function(::Type{Clayton_cop_rev})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      if n > 2
        testθ(θ, "clayton")
      else
        (θ >= -1.) & (θ != 0.) || throw(DomainError("bivariate Clayton not supported for θ < -1 or θ = 0"))
      end
      new(n, θ)
  end
  function(::Type{Clayton_cop_rev})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "clayton", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::Clayton_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations form the Clayton _cop _rev(n, θ) - the reversed Clayton copula (reversed means u → 1 .- u)

```jldoctest

  julia> Random.seed!(43);

  julia> simulate_copula(2, Clayton_cop_rev(2, -0.5))
  2×2 Array{Float64,2}:
   0.819025  0.0922652
   0.224623  0.127926
```
"""
function simulate_copula(t::Int, copula::Clayton_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)
  n = copula.n
  θ = copula.θ
  1 .- simulate_copula(t, Clayton_cop(n, θ); rng = rng)
end

"""
    AMH_cop

Fields:
- n::Int - number of marginals
- θ::Float64 - parameter

Constructor

    AMH_cop(n::Int, θ::Float64)

The Ali-Mikhail-Haq copula parameterized by θ, domain: θ ∈ (0, 1) for n > 2 and  θ ∈ [-1, 1] for n = 2.

Constructor

    AMH_cop(n::Int, θ::Float64, cor::String)

uses cor == "Spearman", "Kendall" to compute θ. Such correlations must be grater than zero and limited from above
due to the θ domain.
            - Spearman correlation must be in range (0, 0.5)
            - Kendall correlation must be in range (0, 1/3)

```jldoctest
julia> AMH_cop(4, .3)
AMH_cop(4, 0.3)

julia> AMH_cop(4, .3, "Kendall")
AMH_cop(4, 0.9999)

```
"""
struct AMH_cop
  n::Int
  θ::Float64
  function(::Type{AMH_cop})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      if n > 2
        testθ(θ, "amh")
      else
      1 >=  θ >= -1 || throw(DomainError("bivariate AMH not supported for θ > 1 or θ < -1"))
      end
      new(n, θ)
  end
  function(::Type{AMH_cop})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "amh", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::AMH_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the Ali-Mikhail-Haq - copulaAMH_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(4, AMH_cop(2, -0.5))
4×2 Array{Float64,2}:
 0.180975  0.477109
 0.775377  0.885537
 0.888934  0.759717
 0.924876  0.313789
```
"""
function simulate_copula(t::Int, copula::AMH_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  n = copula.n
  θ = copula.θ
  if (θ in [0,1]) | (n == 2)*(θ < 0)
    return simulate_copula(t, Chain_of_Archimedeans([θ], "amh"); rng = rng)
  else
    U = zeros(t, n)
    for j in 1:t
      u = rand(rng, n+1)
      U[j,:] = amh_gen(u, θ)
    end
    return U
  end
end

"""
    AMH_cop_rev

Fields:
  - n::Int - number of marginals
  - θ::Float64 - parameter

Constructor

    AMH_cop_rev(n::Int, θ::Float64)

The reversed Ali-Mikhail-Haq copula parametrized by θ, i.e.
such that the output is 1 .- u, where u is modelled by the corresponding AMH copula.
Domain: θ ∈ (0, 1) for n > 2 and  θ ∈ [-1, 1] for n = 2.

Constructor

    AMH_cop_rev(n::Int, θ::Float64, cor::String)

uses cor == "Spearman", "Kendall" to compute θ. Such correlations must be grater than zero and limited from above
due to the θ domain.
              - Spearman correlation must be in range (0, 0.5)
              -  Kendall correlation must be in range (0, 1/3)

```jldoctest
julia> AMH_cop_rev(4, .3)
AMH_cop_rev(4, 0.3)
```
"""
struct AMH_cop_rev
  n::Int
  θ::Float64
  function(::Type{AMH_cop_rev})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      if n > 2
        testθ(θ, "amh")
      else
        1 >=  θ >= -1 || throw(DomainError("bivariate AMH not supported for θ > 1 or θ < -1"))
      end
      new(n, θ)
  end
  function(::Type{AMH_cop_rev})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "amh", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::AMH_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the reversed Ali-Mikhail-Haq copulaAMH _cop _rev(n, θ), reversed means u → 1 .- u.

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(4, rev_amh(2, -0.5))
4×2 Array{Float64,2}:
0.819025  0.522891
0.224623  0.114463
0.111066  0.240283
0.075124  0.686211
```
"""
function simulate_copula(t::Int, copula::AMH_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)
  n = copula.n
  θ = copula.θ
  1 .- simulate_copula(t, AMH_cop(n, θ); rng = rng)
end

"""
    Frank_cop

Fields:
- n::Int - number of marginals
- θ::Float64 - parameter

Constructor

    Frank_cop(n::Int, θ::Float64)

The Frank n variate copula parameterized by θ::Float64.
Domain: θ ∈ (0, ∞) for n > 2 and θ ∈ (-∞, 0) ∪ (0, ∞) for n = 2,
supported for n::Int ≧ 2.

Constructor

    Frank_cop(n::Int, θ::Float64, cor::String)
uses cor == "Spearman", "Kendall" to compute θ, correlations must be greater than zero.

```jldoctest
julia> Frank_cop(2, -5.)
Frank_cop(2, -5.0)

julia> Frank_cop(4, .3)
Frank_cop(4, 0.3)
```
"""
struct Frank_cop
  n::Int
  θ::Float64
  function(::Type{Frank_cop})(n::Int, θ::Float64)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      if n > 2
        testθ(θ, "frank")
      else
        θ != 0 || throw(DomainError("bivariate frank not supported for θ = 0"))
      end
      new(n, θ)
  end
  function(::Type{Frank_cop})(n::Int, ρ::Float64, cor::String)
      n >= 2 || throw(DomainError("not supported for n < 2"))
      θ = getθ4arch(ρ, "frank", cor)
      new(n, θ)
  end
end

"""
    simulate_copula(t::Int, copula::Frank_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the n-variate Frank copula - Frank _cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(4, Frank_cop(2, 3.5))
4×2 Array{Float64,2}:
  0.227231  0.363146
  0.94705   0.979777
  0.877446  0.824164
  0.64929   0.140499

julia> Random.seed!(43);

julia> simulate_copula(4, Frank_cop(2, 0.2, "Spearman"))
  4×2 Array{Float64,2}:
  0.111685  0.277792
  0.92239   0.97086
  0.894941  0.840751
  0.864546  0.271543
```
"""
function simulate_copula(t::Int, copula::Frank_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  n = copula.n
  θ = copula.θ
  if (n == 2) & (θ < 0)
    return simulate_copula(t, Chain_of_Archimedeans([θ], "frank"); rng = rng)
  else
    w = logseriescdf(1-exp(-θ))
    U = zeros(t, n)
    for j in 1:t
      u = rand(rng, n+1)
      U[j,:] = frank_gen(u, θ, w)
    end
    return U
  end
end

"""
  function testθ(θ::Float64, copula::String)

Tests the parameter θ value for archimedean copula, returns void
"""
function testθ(θ::Float64, copula::String)
  if copula == "gumbel"
    θ >= 1 || throw(DomainError("gumbel copula not supported for θ < 1"))
  elseif copula == "amh"
    1 > θ > 0 || throw(DomainError("amh multiv. copula supported only for 0 < θ < 1"))
  else
    θ > 0 || throw(DomainError("generaton not supported for θ ≤ 0"))
  end
  Nothing
end

"""
  useρ(ρ::Float64, copula::String)

Tests the available Spearman correlation for the Archimedean copula.

Returns Float64, the copula parameter θ with the Spearman correlation ρ.
```jldoctest

julia> useρ(0.75, "gumbel")
2.294053859606698
```

"""
function useρ(ρ::Float64, copula::String)
  0 < ρ < 1 || throw(DomainError("Spearman correlation coeficiant must fulfill 0 < ρ < 1"))
  if copula == "amh"
    0 < ρ < 0.5 || throw(DomainError("Spearman correlation coeficiant must fulfill 0 < ρ < 0.5"))
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
  0 < τ < 1 || throw(DomainError("Kendall correlation coeficiant must fulfill 0 < τ < 1"))
  if copula == "amh"
    0 < τ < 1/3 || throw(DomainError("Kendall correlation coeficiant must fulfill 0 < τ < 1/3"))
  end
  τ2θ(τ, copula)
end

"""
  getθ4arch(ρ::Float64, copula::String, cor::String)

  get the copula parameter given the correlation, test if the parameter is in range


```jldoctest

julia> getθ4arch(0.5, "gumbel", "Spearman")
1.541070420842913


julia> getθ4arch(0.5, "gumbel", "Kendall")
2.0



julia> getθ4arch(1.5, "gumbel", "Pearson")
ERROR: AssertionError: Pearson correlation not supported use Kendall or Spearman

```
"""
function getθ4arch(ρ::Float64, copula::String, cor::String)
  if cor == "Spearman"
    θ = useρ(ρ , copula)
  elseif cor == "Kendall"
    θ = useτ(ρ , copula)
  else
    throw(AssertionError("$(cor) correlation not supported use Kendall or Spearman"))
  end
  return θ
end
