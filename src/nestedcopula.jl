"""
  g2tsubcopula!(z::Matrix{Float}, cormat::Matrix{Float}, subn::Array{Int})

Changes data generated using gaussian copula to data generated using student
 subcopula at indices subn.
"""

function g2tsubcopula!(z::Matrix{Float64}, cormat::Matrix{Float64}, subn::Array{Int}, nu::Int = 10)
  d = Chisq(nu)
  U = rand(d, size(z, 1))
  p = TDist(nu)
  for i in subn
    w = quantile(Normal(0, cormat[i,i]), z[:,i])
    z[:,i] = cdf(p, w.*sqrt.(nu./U))
  end
end


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

julia> nestedarchcopulagen(10, [2,2], [2., 3.], 1.1, "clayton")
10×4 Array{Float64,2}:
 0.508525  0.77119   0.998605   0.794352
 0.81838   0.402435  0.73695    0.902096
 0.893028  0.422345  0.993419   0.77742
 0.780068  0.217094  0.0967999  0.151466
 0.181468  0.121059  0.320481   0.251282
 0.896416  0.743234  0.777731   0.678068
 0.719641  0.186449  0.290359   0.48974
 0.466004  0.256329  0.972178   0.892362
 0.389778  0.435092  0.123796   0.097651
 0.930812  0.802301  0.787261   0.49137

```
"""

function nestedarchcopulagen(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64,
                                                                        copula::String, m::Int = 0)
  copula in ["clayton", "amh", "frank", "gumbel"] || throw(AssertionError("$(copula) copula is not supported"))
  nestedcopulag(copula, n, ϕ, θ, rand(t, sum(n)+m+1))
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

function nestedcopulag(copula::String, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, r::Matrix{Float64})
  testnestedθϕ(n, ϕ, θ, copula)
  V0 = getV0(θ, r[:,end], copula)
  X = nestedstep(copula, r[:,1:n[1]], V0, ϕ[1], θ)
  cn = cumsum(n)
  for i in 2:length(n)
    u = r[:,cn[i-1]+1:cn[i]]
    X = hcat(X, nestedstep(copula, u, V0, ϕ[i], θ))
  end
  X = hcat(X, r[:,sum(n)+1:end-1])
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
  u
end

"""
  nestedarchcopulagen::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}},
                                                    ϕ::Vector{Float64}, θ₀::Float64,
                                                    copula::String = "gumbel")

Returns t realisations of ∑ᵢ ∑ⱼ nᵢⱼ variate data of double nested gumbel copula.
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
  θ <= minimum(ϕ) || throw(AssertionError("wrong heirarchy of parameters"))
  X = nestedarchcopulagen(t, n[1], Ψ[1], ϕ[1]./θ, copula)
  for i in 2:length(n)
    X = hcat(X, nestedarchcopulagen(t, n[i], Ψ[i], ϕ[i]./θ, copula))
  end
  phi(-log.(X)./levygen(θ, rand(t)), θ, copula)
end


"""
  nestedarchcopulagen(t::Int, θ::Vector{Float64}, copula::String = "gumbel")

Returns t realisations of length(θ)+1 variate data of hierarchically nested Gumbel copula.
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
  issorted(θ; rev=true) || throw(AssertionError("wrong heirarchy of parameters"))
  θ = vcat(θ, [1.])
  X = rand(t,1)
  for i in 1:length(θ)-1
    X = nestedstep("gumbel", hcat(X, rand(t)), ones(t), θ[i], θ[i+1])
  end
  X
end

"""
  testnestedpars(n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String)

Tests parameters, its hierarchy and size of parametes vector for nested archimedean copulas.
"""

function testnestedθϕ(n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String)
  testθ(θ, copula)
  map(p -> testθ(p, copula), ϕ)
  θ <= minimum(ϕ) || throw(AssertionError("wrong heirarchy of parameters"))
  length(n) == length(ϕ) || throw(AssertionError("number of subcopulas ≠ number of parameters"))
end

# Nested frechet familly copulas

"""
  nestedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))

Retenares data from nested hierarchical frechet copula with parameters
vectors α and β, such that ∀ᵢ 0 α[i] + β[i] ≤1 α[i] > 0, and β[i] > 0 |α| = |β|

```jldoctest
julia> srand(43)

julia> julia> nestedfrechetcopulagen(10, [0.6, 0.4], [0.3, 0.5])
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

function nestedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))
  length(α) == length(β) || throw(AssertionError("different lengths of parameters"))
  minimum(α) >= 0 || throw(AssertionError("negative α parameter"))
  minimum(β) >= 0 || throw(AssertionError("negative β parameter"))
  maximum(α+β) <= 1 || throw(AssertionError("α[i] + β[i] > 0"))
  fncopulagen(t, α, β, rand(t, length(α)+1))
end


function fncopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64}, u::Matrix{Float64})
  p = invperm(sortperm(u[:,1]))
  u = u[:,end:-1:1]
  lx = floor.(Int, t.*α)
  li = floor.(Int, t.*β) + lx
  for j in 1:size(u, 2)-1
    u[p[1:lx[j]],j+1] = u[p[1:lx[j]], j]
    r = p[lx[j]+1:li[j]]
    u[r,j+1] = 1-u[r,j]
  end
  u
end

# copula mix


function copulamix(t::Int, Σ::Matrix{Float64}, inds::Vector{Pair{String,Vector{Int64}}};
                                                λ::Vector{Float64} = [0.8, 0.1],
                                                ν::Int = 10, a::Vector{Float64} = [0.1])
  x = transpose(rand(MvNormal(Σ),t))
  xgauss = copy(x)
  x = cdf(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = norm2unifind(xgauss, Σ, makeind(xgauss, p))
    if p[1] == "Marshal-Olkin"
      map = collect(combinations(1:length(ind),2))
      ρ = [Σ[ind[k[1]], ind[k[2]]] for k in map]
      τ = [moρ2τ(r) for r in ρ]
      x[:,ind] = mocopula(v, length(ind), τ2λ(τ, λ))
    elseif p[1] == "frechet"
      l = length(ind)-1
      α, β = frechetρ2αβ([Σ[ind[k], ind[k+1]] for k in 1:l], a)
      x[:,ind] =fncopulagen(t, α, β, v)
    elseif p[1] == "t-student"
      g2tsubcopula!(x, Σ, ind, ν)
    elseif length(ind) > 2
      m1, m, n = getcors(xgauss[:,ind])
      ϕ = [ρ2θ(m1[i], p[1]) for i in 1:length(m1)]
      θ = ρ2θ(m, p[1])
      x[:,ind] = nestedcopulag(p[1], [length(s) for s in n], ϕ, θ, v)
    else
      θ = ρ2θ(Σ[ind[1], ind[2]], p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    end
  end
  x
end




"""
  makeind(x::Matrix{Float64}, ind::Pair{String,Vector{Int64}})

Returns multiindex of chosen marginals and those most correlated with chosen marginals.
"""

function makeind(x::Matrix{Float64}, ind::Pair{String,Vector{Int64}})
  i = ind[2]
  if ind[1] == "Marshal-Olkin"
    l = length(ind[2])
    for p in 1+l:2^l-1
      i = vcat(i, findsimilar(transpose(x), i))
    end
  elseif ind[1] in ["gumbel", "clayton", "frank", "amh"]
    i = vcat(i, findsimilar(transpose(x), i))
  end
  i
end

function findsimilar(x::Matrix{Float64}, ind::Vector{Int})
  maxd =Float64[]
  for i in 1:size(x,1)
    if !(i in ind)
      y = vcat(x[ind,:], transpose(x[i,:]))
      push!(maxd, sum(sch.maxdists(sch.linkage(y, "average", "correlation"))))
    else
      push!(maxd, Inf)
    end
  end
  find(maxd .== minimum(maxd))
end


"""
  norm2unifind(x::Matrix{Float64}, Σ::Matrix{Float64}, i::Vector{Int})

Given normaly distributed data x with correlation matrix Σ returns
independent uniformly distributed data based on marginals of x indexed by a given
multiindex i.
"""

function norm2unifind(x::Matrix{Float64}, Σ::Matrix{Float64}, i::Vector{Int})
  a, s = eig(Σ[i,i])
  w = x[:, i]*s./transpose(sqrt.(a))
  w[:, end] = sign(cov(x[:, i[1]], w[:, end]))*w[:, end]
  cdf(Normal(0,1), w)
end

function getclust(x::Matrix{Float64})
  Z=sch.linkage(x, "average", "correlation")
  clusts = sch.fcluster(Z, 1, criterion="maxclust")
  for i in 2:size(x,1)
    b = sch.fcluster(Z, i, criterion="maxclust")
    if minimum([count(b.==j) for j in 1:i]) > 1
      clusts = b
    end
  end
  clusts
end

meanΣ(Σ::Matrix{Float64}) = mean(abs.(Σ[find(tril(Σ-eye(Σ)).!=0)]))

function getcors(x::Matrix{Float64})
  inds = getclust(transpose(x))
  Σ = cor(x)
  m = meanΣ(Σ)
  k = maximum(inds)
  m1 = zeros(k)
  ind = Array{Int}[]
  for i in 1:k
    j = find(inds .==i)
    m1[i] = meanΣ(Σ[j,j])
    push!(ind, j)
  end
  m = (m < minimum(m1))? m: minimum(m1)
  m1, m, ind
end
