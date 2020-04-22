

# nested archimedean copulas
# Algorithms from:
# M. Hofert, `Efficiently sampling nested Archimedean copulas` Computational Statistics and Data Analysis 55 (2011) 57–70
# M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
# McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'. Journal of Statistical Computation and Simulation 78, 567–581.

#Basically we use Alg. 5 of McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'.


"""
  Nested_Clayton_cop

Nested Clayton copula, fields:
  - children::Vector{Clayton_cop}  vector of childer copulas
  - m::Int ≧ 0 - number of additional marginals modelled by the parent copula only
  - θ::Float64 - parameterer of parent copula, domain θ > 0.

Nested Clayton copula is in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, last m variables are modelled by the parent copula only.

Constructior Nested_Clayton_cop(children::Vector{Clayton_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting conditin requires
θ <= minimum(ϕ)

If using Nested_Clayton_cop(children::Vector{Clayton_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ

"""

struct Nested_Clayton_cop
  children::Vector{Clayton_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_Clayton_cop})(children::Vector{Clayton_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "clayton")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      maximum(ϕ) < θ+2*θ^2+750*θ^5 || @warn("θ << ϕ, marginals may not be uniform")
      new(children, m, θ)
  end
  function(::Type{Nested_Clayton_cop})(children::Vector{Clayton_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "clayton", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      maximum(ϕ) < θ+2*θ^2+750*θ^5 || @warn("θ << ϕ, marginals may not be uniform")
      new(children, m, θ)
  end
end


#=
function nested_clayton(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  #println(n1)
  n2 = sum(n)+m
  return nestedcopulag("clayton", n1, ϕ, θ, rand(t, n2+1))
end
=#


"""
  Nested_AMH_cop

Nested Ali-Mikhail-Haq copula, fields:
  - children::Vector{AMH_cop}  vector of childer copulas
  - m::Int ≧ 0 - number of additional marginals modelled by the parent copula only
  - θ::Float64 - parameterer of parent copula, domain θ ∈ (0,1).

Nested Ali-Mikhail-Haq copula is in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, last m variables are modelled by the parent copula only.

Constructior Nested_AMH_cop(children::Vector{AMH_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting conditin requires
θ <= minimum(ϕ)

If using Nested_AMH_cop(children::Vector{AMH_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ

"""

struct Nested_AMH_cop
  children::Vector{AMH_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_AMH_cop})(children::Vector{AMH_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "amh")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, m, θ)
  end
  function(::Type{Nested_AMH_cop})(children::Vector{AMH_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "amh", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, m, θ)
  end
end

#=

function simulate_copula1(t::Int, copula::Nested_AMH_cop)
  m = copula.m
  θ = copula.θ
  children = copula.children
  ϕ = [ch.θ for ch in children]
  n = [ch.n for ch in children]
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("amh", n1, ϕ, θ, rand(t, n2+1))
end


function nested_amh(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("amh", n1, ϕ, θ, rand(t, n2+1))
end
=#
"""
  Nested_Frank_cop

Nested Frank copula, fields:
  - children::Vector{Frank_cop}  vector of childer copulas
  - m::Int ≧ 0 - number of additional marginals modelled by the parent copula only
  - θ::Float64 - parameterer of parent copula, domain θ ∈ (0,∞).

Nested Frank copula is in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, last m variables are modelled by the parent copula only.

Constructior Nested_Frank_cop(children::Vector{Frank_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting conditin requires
θ <= minimum(ϕ)

If using Nested_Frank_cop(children::Vector{Frank_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ

"""

struct Nested_Frank_cop
  children::Vector{Frank_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_Frank_cop})(children::Vector{Frank_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "frank")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, m, θ)
  end
  function(::Type{Nested_Frank_cop})(children::Vector{Frank_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "frank", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, m, θ)
  end
end

"""
  Nested_Gumbel_cop

Nested Gumbel copula, fields:
  - children::Vector{Gumbel_cop}  vector of childer copulas
  - m::Int ≧ 0 - number of additional marginals modelled by the parent copula only
  - θ::Float64 - parameterer of parent copula, domain θ ∈ [1,∞).

Nested Gumbel copula is in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, last m variables are modelled by the parent copula only.

Constructior Nested_Gumbel_cop(children::Vector{Frank_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting conditin requires
θ <= minimum(ϕ)

If using Nested_Gumbel_cop(children::Vector{Gumbel_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ

"""

struct Nested_Gumbel_cop
  children::Vector{Gumbel_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_Gumbel_cop})(children::Vector{Gumbel_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "gumbel")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, m, θ)
  end
  function(::Type{Nested_Gumbel_cop})(children::Vector{Gumbel_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "gumbel", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, m, θ)
  end
end

"""
  function simulate_copula1(t::Int, copula::Nested_Clayton_cop)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using:
Nested Clayton copula, Nested AMH copula, Nested Frank copula or Nested Gumbel copula

```jldoctest

julia> Random.seed!(43);

julia> c1 = Clayton_cop(2, 2.)
Clayton_cop(2, 2.0)

julia> c2 = Clayton_cop(2, 3.)
Clayton_cop(2, 3.0)

julia> cp = Nested_Clayton_cop([c1, c2], 1, 1.1)
Nested_Clayton_cop(Clayton_cop[Clayton_cop(2, 2.0), Clayton_cop(2, 3.0)], 1, 1.1)

julia> simulate_copula1(4, cp)
4×5 Array{Float64,2}:
 0.80125   0.879693  0.849878  0.73245   0.538354
 0.25902   0.408295  0.729322  0.228969  0.064877
 0.967594  0.949726  0.887957  0.684867  0.863298
 0.537306  0.182984  0.399726  0.718501  0.415321


 ```jldoctest

 julia> c1 = AMH_cop(2, .7)
 AMH_cop(2, 0.7)

 julia> c2 = AMH_cop(2, .8)
 AMH_cop(2, 0.8)

 julia> cp = Nested_AMH_cop([c1, c2], 1, 0.2)
 Nested_AMH_cop(AMH_cop[AMH_cop(2, 0.7), AMH_cop(2, 0.8)], 1, 0.2)

 julia> Random.seed!(43);

 julia> simulate_copula1(4, cp)
 4×5 Array{Float64,2}:
  0.589196  0.74137   0.748553  0.535984  0.220268
  0.820417  0.928427  0.96363   0.293954  0.0232534
  0.952909  0.926609  0.825948  0.469617  0.767546
  0.958157  0.645533  0.17928   0.719127  0.820758

 ```

```
"""

function simulate_copula1(t::Int, copula::Union{Nested_Clayton_cop, Nested_AMH_cop, Nested_Frank_cop, Nested_Gumbel_cop})
    m = copula.m
    θ = copula.θ
    children = copula.children
    ϕ = [ch.θ for ch in children]
    n = [ch.n for ch in children]
    n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
    n2 = sum(n)+m
    if typeof(copula) == Nested_Clayton_cop
      return nestedcopulag("clayton", n1, ϕ, θ, rand(t, n2+1))
    elseif typeof(copula) == Nested_AMH_cop
      return nestedcopulag("amh", n1, ϕ, θ, rand(t, n2+1))
    elseif typeof(copula) == Nested_Frank_cop
      return nestedcopulag("frank", n1, ϕ, θ, rand(t, n2+1))
    else
      return nestedcopulag("gumbel", n1, ϕ, θ, rand(t, n2+1))
    end
end

#=
function nested_frank(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("frank", n1, ϕ, θ, rand(t, n2+1))
end
=#

"""
    Double_Nested_Gumbel_cop

Double_Nested Gumbel copula, fields:
  - children::Vector{Nested_Gumbel_cop}  vector of childer copulas
  - θ::Float64 - parameterer of parent copula, domain θ ∈ [1,∞).

Requires sufficient nesting condition for θ and child copulas

 If using Doulbe_Nested_Gumbel_cop(children::Vector{Nested_Gumbel_cop}, m::Int, θ::Float64, cor::String)
 uses "Spearman" or "Kendall" correlation to compute θ
"""

struct Double_Nested_Gumbel_cop
  children::Vector{Nested_Gumbel_cop}
  θ::Float64
  function(::Type{Double_Nested_Gumbel_cop})(children::Vector{Nested_Gumbel_cop}, θ::Float64)
      testθ(θ, "gumbel")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, θ)
  end
  function(::Type{Double_Nested_Gumbel_cop})(children::Vector{Nested_Gumbel_cop}, ρ::Float64, cor::String)
      θ = getθ4arch(ρ, "gumbel", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
      new(children, θ)
  end
end

"""
    simulate_copula1(t::Int, copula::Double_Nested_Gumbel_cop)

Simylated double nested Gumbel copula
"""

function simulate_copula1(t::Int, copula::Double_Nested_Gumbel_cop)
    θ = copula.θ
    v = copula.children
    n = [ch.n for ch in v[1].children]
    Ψ = [ch.θ for ch in v[1].children]
    X = nested_gumbel(t, n, Ψ, v[1].θ./θ, v[1].m)
    for i in 2:length(v)
        n = [ch.n for ch in v[i].children]
        Ψ = [ch.θ for ch in v[i].children]
        X = hcat(X, nested_gumbel(t, n, Ψ, v[i].θ./θ, v[i].m))
    end
    phi(-log.(X)./levygen(θ, rand(t)), θ, "gumbel")
end


"""
  nested_gumbel(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Sample nested Gumbel copula, axiliary function for simulate_copula1(t::Int, copula::Double_Nested_Gumbel_cop)
"""

function nested_gumbel(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("gumbel", n1, ϕ, θ, rand(t, n2+1))
end

#=

function nested_gumbel(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}},
                                                             ϕ::Vector{Float64}, θ::Float64)
  θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
  X = nested_gumbel(t, n[1], Ψ[1], ϕ[1]./θ)
  for i in 2:length(n)
    X = hcat(X, nested_gumbel(t, n[i], Ψ[i], ϕ[i]./θ))
  end
  phi(-log.(X)./levygen(θ, rand(t)), θ, "gumbel")
end
=#

"""
    Hierarchical_Gumbel_cop

The hierarchically nested Gumbel copula.
    C_θₙ₋₁(C_θₙ₋₂( ... C_θ₂(C_θ₁(u₁, u₂), u₃)...uₙ₋₁) uₙ)

Fields:
    - n::Int - number of marginals
    - θ::Vector{Float64} - vector of parameters, must be decreasing  and θ[end] ≧ 1, for the
        sufficient nesting condition to be fulfilled.

Costructor
    Hierarchical_Gumbel_cop(θ::Vector{Float64})

    Hierarchical_Gumbel_cop(ρ::Vector{Float64}, cor::String)

    uses cor = "Kendall" or "Spearman" correlation to compute θ

```jldoctest

julia> c = Hierarchical_Gumbel_cop([5., 4., 3.])
Hierarchical_Gumbel_cop(4, [5.0, 4.0, 3.0])

julia> c = Hierarchical_Gumbel_cop([0.95, 0.5, 0.05], "Kendall")
Hierarchical_Gumbel_cop(4, [19.999999999999982, 2.0, 1.0526315789473684])


```

"""

struct Hierarchical_Gumbel_cop
  n::Int
  θ::Vector{Float64}
  function(::Type{Hierarchical_Gumbel_cop})(θ::Vector{Float64})
      testθ(θ[end], "gumbel")
      issorted(θ; rev=true) || throw(DomainError("parameters must be descending"))
      new(length(θ)+1, θ)
  end
  function(::Type{Hierarchical_Gumbel_cop})(ρ::Vector{Float64}, cor::String)
      θ = map(i -> getθ4arch(ρ[i], "gumbel", cor), 1:length(ρ))
      issorted(θ; rev=true) || throw(DomainError("parameters must be descending"))
      new(length(θ)+1, θ)
  end
end

"""
    simulate_copula1(t::Int, copula::Hierarchical_Gumbel_cop)

Returns t realisations of multivariate data from hierarchically nested Gumbel copula, i.e.
Hierarchical_Gumbel_cop(θ)

```jldoctest
julia> using Random

julia> Random.seed!(43);

julia> c = Hierarchical_Gumbel_cop([5., 4., 3.])
Hierarchical_Gumbel_cop(4, [5.0, 4.0, 3.0])

julia> simulate_copula1(3, c)
3×4 Array{Float64,2}:
 0.63944   0.785665  0.646324  0.834632
 0.794524  0.743891  0.638179  0.779129
 0.355646  0.374227  0.119397  0.341991

```
"""

function simulate_copula1(t::Int, copula::Hierarchical_Gumbel_cop)
  θ = copula.θ
  n = copula.n
  θ = vcat(θ, [1.])
  X = rand(t,1)
  for i in 1:(n-1)
    X = nestedstep("gumbel", hcat(X, rand(t)), ones(t), θ[i], θ[i+1])
  end
  X
end

#=
function nested_gumbel(t::Int, θ::Vector{Float64})
  testθ(θ[end], "gumbel")
  issorted(θ; rev=true) || throw(DomainError("wrong heirarchy of parameters"))
  θ = vcat(θ, [1.])
  X = rand(t,1)
  for i in 1:length(θ)-1
    X = nestedstep("gumbel", hcat(X, rand(t)), ones(t), θ[i], θ[i+1])
  end
  X
end
=#

"""
  nestedcopulag(copula::String, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, r::Matrix{Float64})

Given [0,1]ᵗˣˡ ∋ r , returns t realisations of l-1 variate data from nested archimedean copula


```jldoctest
julia> Random.seed!(43)

julia> nestedcopulag("clayton", [2, 2], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
2×4 Array{Float64,2}:
 0.193949  0.230553  0.515404  0.557686
 0.712034  0.761276  0.190189  0.208867
 ```
"""
function nestedcopulag(copula::String, n::Vector{Vector{Int}}, ϕ::Vector{Float64}, θ::Float64,
                                                        r::Matrix{Float64})
  testnestedθϕ(map(length, n), ϕ, θ, copula)
  V0 = getV0(θ, r[:,end], copula)
  X = r[:,1:end-1]
  X[:,n[1]] = nestedstep(copula, r[:,n[1]], V0, ϕ[1], θ)
  for i in 2:length(n)
    X[:,n[i]] = nestedstep(copula, r[:,n[i]], V0, ϕ[i], θ)
  end
  phi(-log.(X)./V0, θ, copula)
end

"""
  nestedstep(copula::String, u::Matrix{Float64}, V0::Union{Vector{Float64}, Vector{Int}}, ϕ::Float64, θ::Float64)

Given u ∈ [0,1]ᵗⁿ and V0 ∈ ℜᵗ returns u ∈ [0,1]ᵗⁿ for a given archimedean nested copula with
inner copulas parameters ϕ anu auter copula parameter θ

```jldoctest
julia> nestedstep("clayton", [0.2 0.8; 0.1 0.7], [0.2, 0.4], 2., 1.5)
2×2 Array{Float64,2}:
 0.283555  0.789899
 0.322614  0.806915
```
"""
function nestedstep(copula::String, u::Matrix{Float64}, V0::Union{Vector{Float64}, Vector{Int}},
                                                        ϕ::Float64, θ::Float64)
  if copula == "amh"
    w = [quantile(NegativeBinomial(v, (1-ϕ)/(1-θ)), rand()) for v in V0]
    u = -log.(u)./(V0 + w)
    X = ((exp.(u) .-ϕ) .*(1-θ) .+θ*(1-ϕ)) ./(1-ϕ)
    return X.^(-V0)
  elseif copula == "frank"
    u = -log.(u)./nestedfrankgen(ϕ, θ, V0)
    X = (1 .-(1 .-exp.(-u)*(1-exp(-ϕ))).^(θ/ϕ))./(1-exp(-θ))
    return X.^V0
  elseif copula == "clayton"
    u = -log.(u)./tiltedlevygen(V0, ϕ/θ)
    return exp.(V0.-V0.*(1 .+u).^(θ/ϕ))
  elseif copula == "gumbel"
    u = -log.(u)./levygen(ϕ/θ, rand(length(V0)))
    return exp.(-u.^(θ/ϕ))
  end
  throw(AssertionError("$(copula) not supported"))
end


"""
  testnestedθϕ(n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String)

Tests parameters, its hierarchy and size of parametes vector for nested archimedean copulas.
"""
function testnestedθϕ(n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String)
  testθ(θ, copula)
  map(p -> testθ(p, copula), ϕ)
  θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
  length(n) == length(ϕ) || throw(AssertionError("number of subcopulas ≠ number of parameters"))
  (copula != "clayton") | (maximum(ϕ) < θ+2*θ^2+750*θ^5) || warn("θ << ϕ for clayton nested copula, marginals may not be uniform")
end
