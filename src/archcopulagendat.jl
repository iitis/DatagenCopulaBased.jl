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
  arch_gen(copula::String, r::Matrix{Float}, θ::Union{Float64, Int})

Auxiliary function used to generate data from archimedean copula (clayton, gumbel, frank or amh)
parametrised by a single parameter θ given a matrix of independent [0,1] distributerd
random vectors.

```jldoctest
  julia> arch_gen("clayton", [0.2 0.6 0.9; 0.4 0.5 0.8], 2.)
  2×2 Array{Float64,2}:
   0.675778  0.851993
   0.687482  0.736394
```

"""
function arch_gen(copula::String, r::Matrix{T}, θ::Union{Float64, Int}) where T <:AbstractFloat
  u = -log.(r[:,1:end-1])./getV0(θ, r[:,end], copula)
  phi(u, θ, copula)
end

"""
  struct Gumbel_cop

    constructor Gumbel_cop(n::Int, θ::Union{Float64, Int})

The Gumbel n variate copula parametrised by θ::Union{Float64, Int} ∈ [1, ∞), supported for n::Int ≧ 2.

If Gumbel_cop(n::Int, θ::Union{Float64, Int}, cor::String) and cor == "Spearman", "Kendall",
uses these correlations to compute θ, correlations must be greater than zero.


"""
struct Gumbel_cop
  n::Int
  θ::Union{Float64, Int}
  function(::Type{Gumbel_cop})(n::Int, θ::Union{Float64, Int})
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      testθ(θ, "gumbel")
      new(n, θ)
  end
  function(::Type{Gumbel_cop})(n::Int, ρ::Union{Float64, Int}, cor::String)
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      θ = getθ4arch(ρ, "gumbel", cor)
      new(n, θ)
  end
end

"""

  simulate_copula1(t::Int, copula::Gumbel_cop)


Returns: t x n Matrix{Float}, t realisations of n-variate data generated from
Gumbel_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula1(2, Gumbel_cop(3, 1.5))
2×3 Array{Float64,2}:
 0.535534  0.900389  0.666363
 0.410877  0.667139  0.637826
 ```

"""
function simulate_copula1(t::Int, copula::Gumbel_cop)
    θ = copula.θ
    n = copula.n
    return arch_gen("gumbel", rand(t,n+1), θ)
end

#=
function gumbel(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")
    θ = getθ4arch(θ, "gumbel", cor)
    return arch_gen("gumbel", rand(t,n+1), θ)
end
=#

"""
  struct Gumbel_cop_rev

constructor Gumbel_cop_rev(n::Int, θ::Union{Float64, Int})

The reversed Gumbel n variate copula, i.e. such that the output is 1 .- u,
where u is modelled by the corresponding Gumbel copula.

Parametrised by θ::Union{Float64, Int} ∈ [1, ∞), supported for n::Int ≧ 2.

If Gumbel_cop_rev(n::Int, θ::Union{Float64, Int}, cor::String),
and cor == "Spearman", "Kendall", uses these correlations to compute θ,
correlations must be greater than zero.


"""
struct Gumbel_cop_rev
  n::Int
  θ::Union{Float64, Int}
  function(::Type{Gumbel_cop_rev})(n::Int, θ::Union{Float64, Int})
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      testθ(θ, "gumbel")
      new(n, θ)
  end
  function(::Type{Gumbel_cop_rev})(n::Int, ρ::Union{Float64, Int}, cor::String)
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      θ = getθ4arch(ρ, "gumbel", cor)
      new(n, θ)
  end
end

"""

  simulate_copula1(t::Int, copula::Gumbel_cop_tev)


Returns: t x n Matrix{Float}, t realisations of n-variate data generated from
Gumbel_cop_rev(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula1(2, Gumbel_cop_rev(3, 1.5))
2×3 Array{Float64,2}:
 0.464466  0.0996114  0.333637
 0.589123  0.332861   0.362174

 ```

"""
function simulate_copula1(t::Int, copula::Gumbel_cop_rev)
    θ = copula.θ
    n = copula.n
    return 1 .- arch_gen("gumbel", rand(t,n+1), θ)
end

#=
function rev_gumbel(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")
  1 .- gumbel(t, n, θ; cor = cor)
end
=#

"""
  struct Clayton_cop

    constructor Clayton_cop(n::Int, θ::Union{Float64, Int})

The Clayton n variate copula parametrised by θ::Union{Float64, Int}.
θ. Domain: θ ∈ (0, ∞) for n > 2 and θ ∈ [-1, 0) ∪ (0, ∞) for n = 2,
supported for n::Int ≧ 2.

If Clayton_cop(n::Int, θ::Union{Float64, Int}, cor::String)
and cor == "Spearman", "Kendall", these correlations are used to compute θ,
correlations must be greater than zero.


"""
struct Clayton_cop
  n::Int
  θ::Union{Float64, Int}
  function(::Type{Clayton_cop})(n::Int, θ::Union{Float64, Int})
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      if n > 2
        testθ(θ, "clayton")
      else
        (θ >= -1.) & (θ != 0.) || throw(DomainError("bivariate Clayton not supported for θ < -1 or θ = 0"))
      end
      new(n, θ)
  end
  function(::Type{Clayton_cop})(n::Int, ρ::Union{Float64, Int}, cor::String)
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      θ = getθ4arch(ρ, "clayton", cor)
      new(n, θ)
  end
end

"""

  simulate_copula1(t::Int, copula::Clayton_cop)


Returns: t x n Matrix{Float}, t realisations of n-variate data generated from
Clayton_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula1(10, Clayton_cop(2, 1))
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

 julia> simulate_copula1(2, Clayton_cop(2, -0.5))
 2×2 Array{Float64,2}:
  0.180975  0.907735
  0.775377  0.872074

 ```

"""
function simulate_copula1(t::Int, copula::Clayton_cop)
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

  constructor Clayton_cop_rev(n::Int, θ::Union{Float64, Int})

The reversed Clayton n variate copula parametrised by θ::Union{Float64, Int} i.e.
such that the output is 1 .- u, where u is modelled by the corresponding Clayton copula.
θ. Domain: θ ∈ (0, ∞) for n > 2 and θ ∈ [-1, 0) ∪ (0, ∞) for n = 2,
supported for n::Int ≧ 2.

If Clayton_cop_rev(n::Int, θ::Union{Float64, Int}, cor::String)
where cor == "Spearman", "Kendall", these correlations are used to compute θ.
Correlations must be greater than zero.
"""


struct Clayton_cop_rev
  n::Int
  θ::Union{Float64, Int}
  function(::Type{Clayton_cop_rev})(n::Int, θ::Union{Float64, Int})
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      if n > 2
        testθ(θ, "clayton")
      else
        (θ >= -1.) & (θ != 0.) || throw(DomainError("bivariate Clayton not supported for θ < -1 or θ = 0"))
      end
      new(n, θ)
  end
  function(::Type{Clayton_cop_rev})(n::Int, ρ::Union{Float64, Int}, cor::String)
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      θ = getθ4arch(ρ, "clayton", cor)
      new(n, θ)
  end
end

"""
  rev_clayton(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")

  Returns 1 .- clayton(....)

```jldoctest

  julia> Random.seed!(43);

  julia> rev_clayton(2, 2, -0.5)
  2×2 Array{Float64,2}:
   0.819025  0.0922652
   0.224623  0.127926

```

"""

function simulate_copula1(t::Int, copula::Clayton_cop_rev)
  n = copula.n
  θ = copula.θ
  1 .- simulate_copula1(t, Clayton_cop(n, θ))
end

"""
  AMH_cop

  constructor AMH_cop(n::Int, θ::Union{Float64, Int})

The Ali-Mikhail-Haq copula parametrised by θ. Domain: θ ∈ (0, 1) for n > 2 and  θ ∈ [-1, 1] for n = 2.

Using constructor AMH_cop(n::Int, θ::Union{Float64, Int}, cor::String)
where cor == "Spearman", "Kendall", these correlations are used to compute θ.
Such correlations must be grater than zero and limited from above
due to the θ domain.
              - Spearman correlation must be in range (0, 0.5)
              -  Kendall correlation must be in range (0, 1/3)
"""

struct AMH_cop
  n::Int
  θ::Union{Float64, Int}
  function(::Type{AMH_cop})(n::Int, θ::Union{Float64, Int})
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      if n > 2
        testθ(θ, "amh")
      else
      1 >=  θ >= -1 || throw(DomainError("bivariate AMH not supported for θ > 1 or θ < -1"))
      end
      new(n, θ)
  end
  function(::Type{AMH_cop})(n::Int, ρ::Union{Float64, Int}, cor::String)
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      θ = getθ4arch(ρ, "amh", cor)
      new(n, θ)
  end
end

"""

  simulate_copula1(t::Int, copula::AMH_cop)

Returns: t x n Matrix{Float}, t realisations of n-variate Ali-Mikhail-Haq copula


```jldoctest
julia> Random.seed!(43);

julia> simulate_copula1(4, AMH_cop(2, -0.5))
4×2 Array{Float64,2}:
 0.180975  0.477109
 0.775377  0.885537
 0.888934  0.759717
 0.924876  0.313789

 ```

"""

function simulate_copula1(t::Int, copula::AMH_cop)
  n = copula.n
  θ = copula.θ
  if (θ in [0,1]) | (n == 2)*(θ < 0)
    return chaincopulagen(t, [θ], "amh")
  else
    return arch_gen("amh", rand(t,n+1), θ)
  end
end

function amh(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")
  if (θ in [0,1]) | (n == 2)*(θ < 0)
    return chaincopulagen(t, [θ], "amh"; cor=cor)
  else
    if cor == ""
      testθ(θ, "amh")
    else
      θ = getθ4arch(θ, "amh", cor)
    end
    return arch_gen("amh", rand(t,n+1), θ)
  end
end


"""
  AMH_cop_rev

  constructor AMH_cop_rev(n::Int, θ::Union{Float64, Int})

The reversed Ali-Mikhail-Haq copula parametrised by θ, i.e.
such that the output is 1 .- u, where u is modelled by the corresponding Clayton copula.

Domain: θ ∈ (0, 1) for n > 2 and  θ ∈ [-1, 1] for n = 2.


Using constructor AMH_cop_rev(n::Int, θ::Union{Float64, Int}, cor::String)
where cor == "Spearman", "Kendall", these correlations are used to compute θ.
Such correlations must be grater than zero and limited from above
due to the θ domain.
              - Spearman correlation must be in range (0, 0.5)
              -  Kendall correlation must be in range (0, 1/3)
"""

struct AMH_cop_rev
  n::Int
  θ::Union{Float64, Int}
  function(::Type{AMH_cop_rev})(n::Int, θ::Union{Float64, Int})
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      if n > 2
        testθ(θ, "amh")
      else
        1 >=  θ >= -1 || throw(DomainError("bivariate AMH not supported for θ > 1 or θ < -1"))
      end
      new(n, θ)
  end
  function(::Type{AMH_cop_rev})(n::Int, ρ::Union{Float64, Int}, cor::String)
      n >= 2 || throw(AssertionError("not supported for n < 2"))
      θ = getθ4arch(ρ, "amh", cor)
      new(n, θ)
  end
end

"""

  simulate_copula1(t::Int, copula::AMH_cop_rev)

Returns: t x n Matrix{Float}, t realisations of n-variate reversed
 Ali-Mikhail-Haq copula

```jldoctest
  julia> Random.seed!(43);

  julia> simulate_copula1(4, rev_amh(2, -0.5))
  4×2 Array{Float64,2}:
  0.819025  0.522891
  0.224623  0.114463
  0.111066  0.240283
  0.075124  0.686211

```

"""


function simulate_copula1(t::Int, copula::AMH_cop_rev)
  n = copula.n
  θ = copula.θ
  1 .- simulate_copula1(t, AMH_cop(n, θ))
end

function rev_amh(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")
  return 1 .- amh(t, n, θ; cor = cor)
end

"""

  frank(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from the clayton copula
parametrised by θ. Domain θ ∈ (0, ∞) for n > 2 and θ ∈ (-∞, 0) ∪ (0, ∞) for n = 2.

It cor == "Spearman", "Kendall", uses these correlations for θ (these must be grater than zero).

```jldoctest

julia> Random.seed!(43);

julia> frank(4, 2, 3.5)
4×2 Array{Float64,2}:
 0.227231  0.363146
 0.94705   0.979777
 0.877446  0.824164
 0.64929   0.140499


 julia> Random.seed!(43);

 julia> frank(4, 2, 0.2; cor = "Spearman")
 4×2 Array{Float64,2}:
  0.111685  0.277792
  0.92239   0.97086
  0.894941  0.840751
  0.864546  0.271543

```
"""

function frank(t::Int, n::Int, θ::Union{Float64, Int}; cor::String = "")
  if (n == 2)*(θ < 0)
    return chaincopulagen(t, [θ], "frank"; cor=cor)
  else
    if cor == ""
      testθ(θ, "frank")
    else
      θ = getθ4arch(θ, "frank", cor)
    end
    return arch_gen("frank", rand(t,n+1), θ)
  end
end


"""
  function testθ(θ::Union{Float64, Int}, copula::String)

Tests the parameter θ value for archimedean copula, returns void
"""
function testθ(θ::Union{Float64, Int}, copula::String)
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

Tests the available pearson correlation for archimedean copula, returns Float,
corresponding copula parameter θ.
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
  getθ4arch(ρ::Union{Float64, Int}, copula::String, cor::String)

  get the copula parameter given the correlation, test if the parameter is in range


```jldoctest

julia> getθ4arch(0.5, "gumbel", "Spearman")
1.541070420842913


julia> getθ4arch(0.5, "gumbel", "Kendall")
2.0



julia> getθ4arch(1.5, "gumbel", "Pearson")
ERROR: AssertionError: Pearson correlation not supported

```
"""

function getθ4arch(ρ::Union{Float64, Int}, copula::String, cor::String)
  if cor == "Spearman"
    θ = useρ(ρ , copula)
  elseif cor == "Kendall"
    θ = useτ(ρ , copula)
  else
    throw(AssertionError("$(cor) correlation not supported"))
  end
  return θ
end
