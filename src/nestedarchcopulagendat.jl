

# nested archimedean copulas
# Algorithms from:
# M. Hofert, `Efficiently sampling nested Archimedean copulas` Computational Statistics and Data Analysis 55 (2011) 57–70
# M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
# McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'. Journal of Statistical Computation and Simulation 78, 567–581.

#Basically we use Alg. 5 of McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'.


"""
  function nested_clayton(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested Clayton copula in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).

Parent copula parameter is θ, i'th child copula size is n[i] and parameter is ϕ[i].
If m ≠ 0, last m variables are modelled by the parent copula only.

Sufficient nesting condition yields ∀ᵢ ϕ[i] ≥ θ. Limits for θ are such as as for n > 2 variate Clayton copula.

```jldoctest

    julia> Random.seed!(43);

  julia> nested_clayton(4, [2,2], [2., 3.], 1.1, 1)
  4×5 Array{Float64,2}:
   0.80125   0.879693  0.849878  0.73245   0.538354
   0.25902   0.408295  0.729322  0.228969  0.064877
   0.967594  0.949726  0.887957  0.684867  0.863298
   0.537306  0.182984  0.399726  0.718501  0.415321


```
"""

function nested_clayton(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("clayton", n1, ϕ, θ, rand(t, n2+1))
end

"""
  function nested_amh(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested Ali-Mikhail-Haq copula in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).

Parent copula parameter is θ, i'th child copula size is n[i] and parameter is ϕ[i].
If m ≠ 0, last m variables are modelled by the parent copula only.

Sufficient nesting condition yields ∀ᵢ ϕ[i] ≥ θ. Limits for θ are such as as for n > 2 variate Ali-Mikhail-Haq copula.

```jldoctest

  julia> Random.seed!(43);

  julia> nested_amh(4, [2,2], [.7, .8], 0.2, 1)
  4×5 Array{Float64,2}:
   0.589196  0.74137   0.748553  0.535984  0.220268
   0.820417  0.928427  0.96363   0.293954  0.0232534
   0.952909  0.926609  0.825948  0.469617  0.767546
   0.958157  0.645533  0.17928   0.719127  0.820758


```
"""

function nested_amh(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("amh", n1, ϕ, θ, rand(t, n2+1))
end

"""
  function nested_frank(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested Frank copula in the form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).

Parent copula parameter is θ, i'th child copula size is n[i] and parameter is ϕ[i].
If m ≠ 0, last m variables are modelled by the parent copula only.

Sufficient nesting condition yields ∀ᵢ ϕ[i] ≥ θ. Limits for θ are such as as for n > 2 variate Frank copula.

```jldoctest

  julia> Random.seed!(43);


  julia> nested_frank(4, [2,2], [.7, .8], 0.2, 1)
  4×5 Array{Float64,2}:
  0.716339  0.83584   0.781909   0.564834   0.153479
  0.706925  0.878726  0.934177   0.0671711  0.0262611
  0.91952   0.875725  0.742692   0.277226   0.701578
  0.895148  0.321701  0.0521964  0.654462   0.838012

```
"""

function nested_frank(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("frank", n1, ϕ, θ, rand(t, n2+1))
end

"""
  nested_gumbel(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

this is more complicated, there are 3 posibilities
"""

function nested_gumbel(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("gumbel", n1, ϕ, θ, rand(t, n2+1))
end

"""
  Returns t realisations of ∑ᵢ ∑ⱼ nᵢⱼ variate data from double nested gumbel copula.
  C_θ(C_ϕ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_ϕₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))
   where lᵢ = length(n[i])
"""

function nested_gumbel(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}},
                                                             ϕ::Vector{Float64}, θ::Float64)
  θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
  X = nested_gumbel(t, n[1], Ψ[1], ϕ[1]./θ)
  for i in 2:length(n)
    X = hcat(X, nested_gumbel(t, n[i], Ψ[i], ϕ[i]./θ))
  end
  phi(-log.(X)./levygen(θ, rand(t)), θ, "gumbel")
end


"""
  nested_gumbel(t::Int, θ::Vector{Float64})

Returns t realisations of length(θ)+1 variate data from hierarchically nested Gumbel copula.
C_θₙ(... C_θ₂(C_θ₁(u₁, u₂), u₃)...,  uₙ)


"""
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
