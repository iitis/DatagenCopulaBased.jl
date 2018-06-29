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
    return (1 + u).^(-1/θ)
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
  cor in ["Spearman", "Kendall", ""] || throw(AssertionError("$(cor) correlation not supported"))
  copula in ["clayton", "amh", "frank", "gumbel"] || throw(AssertionError("$(copula) copula is not supported"))
  if (n == 2)*((copula != "gumbel")*(θ < 0) | (copula == "amh")*(θ in [0,1]))
    return chaincopulagen(t, [θ], copula; rev=rev, cor=cor)
  elseif cor == "Spearman"
    θ = useρ(θ , copula)
  elseif cor == "Kendall"
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
    θ >= 1 || throw(DomainError("generaton not supported for θ < 1"))
  elseif copula == "amh"
    1 > θ > 0 || throw(DomainError("amh multiv. copula supported only for 0 < θ < 1"))
  else
    θ > 0 || throw(DomainError("generaton not supported for θ ≤ 0"))
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
  0 < ρ < 1 || throw(DomainError("correlation coeficiant must fulfill 0 < ρ < 1"))
  if copula == "amh"
    0 < ρ < 0.5 || throw(DomainError("correlation coeficiant must fulfill 0 < ρ < 0.5"))
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
  0 < τ < 1 || throw(DomainError("correlation coeficiant must fulfill 0 < τ < 1"))
  if copula == "amh"
    0 < τ < 1/3 || throw(DomainError("correlation coeficiant must fulfill 0 < τ < 1/3"))
  end
  τ2θ(τ, copula)
end
