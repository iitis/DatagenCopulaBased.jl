

# nested archimedean copulas
# Algorithms from:
# M. Hofert, `Efficiently sampling nested Archimedean copulas` Computational Statistics and Data Analysis 55 (2011) 57–70
# M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
# McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'. Journal of Statistical Computation and Simulation 78, 567–581.

"""
  nestedarchcopulagen(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested archimedean copula, outer copula parameter is θ, inner i'th copulas parameter is
ϕ[i] and size is n[i]. If m ≠ 0, last m variables are from outer copula only.

Following copula families are supported: clayton, frank, gumbel and amh --
Ali-Mikhail-Haq.

Nested archimedean copula in in a form C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
Parameters must fulfill ∀ᵢ ϕ[i] ≥ θ

Basically uses Alg. 5 McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'.
Journal of Statistical Computation and Simulation 78, 567–581.

```jldoctest
julia> srand(43);

julia> nestedarchcopulagen(10, [2,2], [2., 3.], 1.1, "clayton", 1)
10×5 Array{Float64,2}:
 0.414567  0.683167   0.9953    0.607738  0.793386
 0.533001  0.190563   0.17076   0.273119  0.78807
 0.572782  0.161307   0.418821  0.110356  0.661781
 0.623807  0.140974   0.295422  0.454368  0.477065
 0.386276  0.266261   0.559423  0.449874  0.294137
 0.219757  0.122586   0.371318  0.298965  0.507315
 0.322658  0.0627113  0.738565  0.919912  0.19471
 0.131938  0.0672061  0.364721  0.220329  0.662842
 0.773414  0.812113   0.639333  0.527118  0.545043
 0.958656  0.871822   0.958339  0.801866  0.862751

```
"""
function nestedarchcopulagen(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  nestedarchcopulagen(t, sum(n)+m, n1, ϕ, θ, copula)
end


function nestedarchcopulagen(t::Int, n::Int, n1::Vector{Vector{Int}}, ϕ::Vector{Float64}, θ::Float64,
                                                     copula::String)
  copula in ["clayton", "amh", "frank", "gumbel"] || throw(AssertionError("$(copula) copula is not supported"))
  nestedcopulag(copula, n1, ϕ, θ, rand(t, n+1))
end

"""
  nestedcopulag(copula::String, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, r::Matrix{Float64})

Given [0,1]ᵗˣˡ ∋ r , returns t realisations of l-1 variate data from nested archimedean copula


```jldoctest
julia> srand(43)

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
    X = ((exp.(u)-ϕ)*(1-θ)+θ*(1-ϕ))/(1-ϕ)
    return X.^(-V0)
  elseif copula == "frank"
    u = -log.(u)./nestedfrankgen(ϕ, θ, V0)
    X = (1-(1-exp.(-u)*(1-exp(-ϕ))).^(θ/ϕ))./(1-exp(-θ))
    return X.^V0
  elseif copula == "clayton"
    u = -log.(u)./tiltedlevygen(V0, ϕ/θ)
    return exp.(V0.-V0.*(1.+u).^(θ/ϕ))
  elseif copula == "gumbel"
    u = -log.(u)./levygen(ϕ/θ, rand(length(V0)))
    return exp.(-u.^(θ/ϕ))
  end
  throw(AssertionError("$(copula) not supported"))
end

"""
  nestedarchcopulagen::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}},
                                                    ϕ::Vector{Float64}, θ₀::Float64,
                                                    copula::String = "gumbel")

Returns t realisations of ∑ᵢ ∑ⱼ nᵢⱼ variate data from double nested gumbel copula.
C_θ(C_ϕ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_ϕₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))
 where lᵢ = length(n[i])

  ```jldoctest
  julia> srand(43)

  julia> x = nestedarchcopulagen(5, [[2,2],[2]], [[3., 2.], [4.]], [1.5, 2.1], 1.2, "gumbel")
  5×6 Array{Float64,2}:
   0.464403  0.711722   0.883035   0.896706   0.888614   0.826514
   0.750596  0.768193   0.0659561  0.0252472  0.996014   0.989127
   0.825211  0.712079   0.581356   0.507739   0.882675   0.84959
   0.276326  0.0827071  0.240836   0.434629   0.0184611  0.031363
   0.208422  0.504727   0.27561    0.639089   0.481855   0.573715

  ```
"""


function nestedarchcopulagen(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}},
                                                             ϕ::Vector{Float64}, θ::Float64,
                                                             copula::String = "gumbel")
  copula == "gumbel" ||
  throw(AssertionError("double nested cop. generator supported only for gumbel familly"))
  θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
  X = nestedarchcopulagen(t, n[1], Ψ[1], ϕ[1]./θ, copula)
  for i in 2:length(n)
    X = hcat(X, nestedarchcopulagen(t, n[i], Ψ[i], ϕ[i]./θ, copula))
  end
  phi(-log.(X)./levygen(θ, rand(t)), θ, copula)
end


"""
  nestedarchcopulagen(t::Int, θ::Vector{Float64}, copula::String = "gumbel")

Returns t realisations of length(θ)+1 variate data from hierarchically nested Gumbel copula.
C_θₙ(... C_θ₂(C_θ₁(u₁, u₂), u₃)...,  uₙ)

```jldoctest
julia> srand(43)

julia> x = nestedarchcopulagen(5, [4., 3., 2.], "gumbel")
5×4 Array{Float64,2}:
 0.483466  0.621572  0.241025  0.312664
 0.827237  0.696634  0.768802  0.730543
 0.401159  0.462126  0.412573  0.72571
 0.970726  0.964746  0.940314  0.934625
 0.684486  0.614142  0.690664  0.401897
```
"""

function nestedarchcopulagen(t::Int, θ::Vector{Float64}, copula::String = "gumbel")
  copula == "gumbel" ||
  throw(AssertionError("hierarchically nasted cop. generator supported only for gumbel familly"))
  testθ(θ[end], copula)
  issorted(θ; rev=true) || throw(DomainError("wrong heirarchy of parameters"))
  θ = vcat(θ, [1.])
  X = rand(t,1)
  for i in 1:length(θ)-1
    X = nestedstep("gumbel", hcat(X, rand(t)), ones(t), θ[i], θ[i+1])
  end
  X
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
