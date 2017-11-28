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
    z[:,i] = cdf.(p, w.*sqrt.(nu./U))
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
  θ <= minimum(ϕ) || throw(AssertionError("wrong heirarchy of parameters"))
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

VP = Vector{Pair{String,Vector{Int64}}}
# copula mix
"""
  copulamix(t::Int, Σ::Matrix{Float}, inds::VP; λ::Vector{Float} = [6., 3., 1., 15.], ν::Int = 2,
                                                a::Vector{Float} = [0.1])

Returns x ∈ [0,1]ᵗⁿ data generated from gaussian copula with given correlation matrix Σ,
and replaced by gumbel, clayton, frank, amh (Ali-Mikhail-Haq), mo ("Marshal-Olkin"),
frechet or t-student copula at given marginal indices.

Thise copulas are indicated in inds = [copulaname::String, indices of marginals::Vector{Int}].
Indices array must be disjoint for different copulas.

Additional copula parameters are supplied as a named parameters, for t-student copula: ν::Int,
for "Marshal-Olkin" λ::Vector{Float64}, for frechet copula a::Vector{Float64} = α - β

```jldoctest

julia> d = ["mo" => [1,2,3], "clayton" => [4,5,6]];

julia> srand(43);

julia> Σ = cormatgen(7);

julia> copulamix(10, Σ, d)
10×7 Array{Float64,2}:
 0.746289   0.77815    0.800142   0.17872    0.415729    0.100993  0.289826
 0.487619   0.540306   0.578558   0.667372   0.854448    0.396435  0.252692
 0.959344   0.653536   0.685165   0.0694169  0.635065    0.943713  0.0463879
 0.949993   0.956981   0.961668   0.0512759  0.00436219  0.20456   0.19945
 0.718963   0.753662   0.777721   0.125815   0.145763    0.408557  0.379778
 0.241307   0.295647   0.338514   0.760856   0.772814    0.165766  0.544807
 0.0223038  0.0383994  0.0551596  0.960242   0.354697    0.381062  0.183945
 0.196893   0.0776412  0.103137   0.426957   0.40978     0.246162  0.831808
 0.856694   0.875834   0.888832   0.430832   0.0839215   0.610538  0.0235287
 0.574165   0.621527   0.655253   0.129092   0.0783403   0.41072   0.202428

```
"""


function copulamix(t::Int, Σ::Matrix{Float64}, inds::VP; λ::Vector{Float64} = [6., 3., 1., 15.],
                                                ν::Int = 2, a::Vector{Float64} = [0.1])
  testind(inds)
  x = transpose(rand(MvNormal(Σ),t))
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = norm2unifind(xgauss, Σ, makeind(xgauss, p))
    if p[1] == "mo"
      length(ind) < 4 || throw(DomainError("not supported for Marshal-Olkin subcopula of number of marginals > 3"))
      map = collect(combinations(1:length(ind),2))
      ρ = [Σ[ind[k[1]], ind[k[2]]] for k in map]
      τ = [moρ2τ(r) for r in ρ]
      x[:,ind] = mocopula(v, length(ind), τ2λ(τ, λ))
    elseif p[1] == "frechet"
      l = length(ind)-1
      α, β = frechetρ2αβ([Σ[ind[k], ind[k+1]] for k in 1:l], a)
      x[:,ind] =fncopulagen(α, β, v)
    elseif p[1] == "t-student"
      g2tsubcopula!(x, Σ, ind, ν)
    elseif length(ind) > 2
      m1, m, n = getcors(xgauss[:,ind])
      ϕ = [ρ2θ(abs.(m1[i]), p[1]) for i in 1:length(m1)]
      θ = ρ2θ(abs.(m), p[1])
      x[:,ind] = nestedcopulag(p[1], [length(s) for s in n], ϕ, θ, v)
    else
      θ = ρ2θ(Σ[ind[1], ind[2]], p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    end
  end
  x
end

"""
  testind(inds::Vector{Pair{String,Vector{Int64}}})

Tests if the sub copula name is supported and if their indices are disjoint.
"""

function testind(inds::Vector{Pair{String,Vector{Int64}}})
  indar = []
  for i in 1:length(inds)
    indar = vcat(indar, inds[i][2])
    inds[i][1] in ["gumbel", "clayton", "frank", "amh", "mo", "t-student", "frechet"] ||
    throw(AssertionError("$(inds[i][1]) copula family not supported"))
  end
  unique(indar) == indar || throw(AssertionError("differnt subcopulas must heve different indices"))
end

"""
  makeind(x::Matrix{Float64}, ind::Pair{String,Vector{Int64}})

Returns multiindex of chosen marginals and those most correlated with chosen marginals.
"""

function makeind(x::Matrix{Float64}, ind::Pair{String,Vector{Int64}})
  i = ind[2]
  if ind[1] == "mo"
    l = length(ind[2])
    for p in 1+l:2^l-1
      i = vcat(i, findsimilar(transpose(x), i))
    end
  elseif ind[1] in ["gumbel", "clayton", "frank", "amh"]
    i = vcat(i, findsimilar(transpose(x), i))
  end
  i
end

"""
  findsimilar(x::Matrix{Float64}, ind::Vector{Int})

Returns Array{Int64,1}, an index of most simillar vector to those indexed by ind from x

```jldoctest

julia> x = [0.1 0.2 0.3 0.4; 0.2 0.3 0.4 0.5; 0.2 0.2 0.4 0.4; 0.1 0.3 0.5 0.6]

julia> findsimilar(x, [1,2])
1-element Array{Int64,1}:
 4
```
"""
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
  cdf.(Normal(0,1), w)
end

"""
  getclust(x::Matrix{Float64})

Returns Array{Int} of that indicates a clusters of marginals given a data matrix

``` jldoctest
julia> srand(43)

julia> getclust(randn(4,100))
4-element Array{Int32,1}:
 1
 1
 1
 1
```
"""

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

"""
  meanΣ(Σ::Matrix{Float64})

Returns Float64, a mean of the mean of lower diagal elements of a matrix
"""
meanΣ(Σ::Matrix{Float64}) = mean(abs.(Σ[find(tril(Σ-eye(Σ)).!=0)]))

"""
  getcors(x::Matrix{Float64})

retruns Float64, Vector{Float64} and Venctor{Vector{Int}}, a general mean correlation of
data, mean correlations in each cluster and indices of clusters.
"""
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
