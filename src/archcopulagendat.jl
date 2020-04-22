# Archimedean copulas
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
function getV0(θ::Float64, v::Vector{Float64}, copula::String)
  if copula == "clayton"
    return quantile.(Gamma(1/θ, 1), v)
  elseif copula == "amh"
    return 1 .+quantile.(Geometric(1-θ), v)
  elseif copula == "frank"
    return logseriesquantile(1-exp(-θ), v)
  elseif copula == "gumbel"
    return levygen(θ, v)
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
function phi(u::Matrix{Float64}, θ::Float64, copula::String)
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
function arch_gen(copula::String, r::Matrix{T}, θ::Float64) where T <:AbstractFloat
  u = -log.(r[:,1:end-1])./getV0(θ, r[:,end], copula)
  phi(u, θ, copula)
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
    simulate_copula(t::Int, copula::Gumbel_cop)

Returns t realizations from the Gumbel copula -  Gumbel_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(2, Gumbel_cop(3, 1.5))
2×3 Array{Float64,2}:
 0.535534  0.900389  0.666363
 0.410877  0.667139  0.637826
```
"""
function simulate_copula(t::Int, copula::Gumbel_cop)
    θ = copula.θ
    n = copula.n
    return arch_gen("gumbel", rand(t,n+1), θ)
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
    simulate_copula(t::Int, copula::Gumbel_cop_rev)

Returns t realizations from the Gumbel _cop _rev(n, θ) - the reversed Gumbel copula (reversed means u → 1 .- u).

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(2, Gumbel_cop_rev(3, 1.5))
2×3 Array{Float64,2}:
 0.464466  0.0996114  0.333637
 0.589123  0.332861   0.362174
```
"""
function simulate_copula(t::Int, copula::Gumbel_cop_rev)
    θ = copula.θ
    n = copula.n
    return 1 .- arch_gen("gumbel", rand(t,n+1), θ)
end

"""
    Clayton_cop

Fields:
  - n::Int - number of marginals
  - θ::Float64 - parameter

Constructor

    Clayton_cop(n::Int, θ::Float64)

The Clayton n variate copula parameterized by θ::Float64, such that θ ∈ (0, ∞) for n > 2 and θ ∈ [-1, 0) ∪ (0, ∞) for n = 2,
supported for n::Int ≧ 2.

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
    simulate_copula(t::Int, copula::Clayton_cop)


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
function simulate_copula(t::Int, copula::Clayton_cop)
    θ = copula.θ
    n = copula.n
  if (n == 2) & (θ < 0)
    return chaincopulagen(t, [θ], "clayton")
  else
    return arch_gen("clayton", rand(t,n+1), θ)
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
    simulate_copula(t::Int, copula::Clayton_cop_rev)

Returns t realizations form the Clayton _cop _rev(n, θ) - the reversed Clayton copula (reversed means u → 1 .- u)

```jldoctest

  julia> Random.seed!(43);

  julia> simulate_copula(2, Clayton_cop_rev(2, -0.5))
  2×2 Array{Float64,2}:
   0.819025  0.0922652
   0.224623  0.127926
```
"""
function simulate_copula(t::Int, copula::Clayton_cop_rev)
  n = copula.n
  θ = copula.θ
  1 .- simulate_copula(t, Clayton_cop(n, θ))
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
    simulate_copula(t::Int, copula::AMH_cop)

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
function simulate_copula(t::Int, copula::AMH_cop)
  n = copula.n
  θ = copula.θ
  if (θ in [0,1]) | (n == 2)*(θ < 0)
    return chaincopulagen(t, [θ], "amh")
  else
    return arch_gen("amh", rand(t,n+1), θ)
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
    simulate_copula(t::Int, copula::AMH_cop_rev)

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
function simulate_copula(t::Int, copula::AMH_cop_rev)
  n = copula.n
  θ = copula.θ
  1 .- simulate_copula(t, AMH_cop(n, θ))
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
    simulate_copula(t::Int, copula::Frank_cop)

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
function simulate_copula(t::Int, copula::Frank_cop)
  n = copula.n
  θ = copula.θ
  if (n == 2) & (θ < 0)
    return chaincopulagen(t, [θ], "frank")
  else
    return arch_gen("frank", rand(t,n+1), θ)
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
