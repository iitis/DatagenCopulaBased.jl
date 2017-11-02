# nested copulas
# Algorithms from M. Hofert, `Efficiently sampling nested Archimedean copulas`
# Computational Statistics and Data Analysis 55 (2011) 57–70

"""
  nestedarchcopulagen(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested archimedean copula, outer copula parameter is θ, inner i'th copulas parameter is
ϕ[i] and size is n[i]. If m ≠ 0, last m variables are from outer copula only, see Alg. 5
McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'. Journal of Statistical
 Computation and Simulation 78, 567–581.
"""


nestedarchcopulagen(copula::String, t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0) =
  nestedcopulag(copula, t, n, ϕ, θ, rand(t, sum(n)+m+1))

  """
    nestedcopulag(copula::String, t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, r::Matrix{Float64})

  Returns t realisations of ∑ᵢ nᵢ variate data of nested archimedean copula
  C_θ(C_Φ₁(u₁₁, ..., u₁,ₙ₁), C_θ(C_Φₖ(uₖ₁, ..., uₖ,ₙₖ)) where k = length(n).

  M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
  """

function nestedcopulag(copula::String, t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, r::Matrix{Float64})
  testnestedpars(θ, ϕ, n)
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
  testnestedpars(θ::Float64, ϕ::Vector{Float64}, n::Vector{Int})

Tests the hierarchy of parameters for the nested archimedean copula where both parent and
childs are from the same family
"""

function testnestedpars(θ::Float64, ϕ::Vector{Float64}, n::Vector{Int})
  θ <= minimum(ϕ) || throw(AssertionError("wrong heirarchy of parameters"))
  length(n) == length(ϕ) || throw(AssertionError("number of subcopulas ≠ number of parameters"))
end

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
  nestedgumbelcopulat::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, Φ::Vector{Float64}, θ₀::Float64)

Returns t realisations of ∑ᵢ ∑ⱼ nᵢⱼ variate data of double nested Gumbel copula.
C_θ(C_Φ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_Φₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))
 where lᵢ = length(n[i])

"""


function nestedgumbelcopula(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, Φ::Vector{Float64}, θ::Float64)
  θ <= minimum(Φ) || throw(AssertionError("wrong heirarchy of parameters"))
  X = nestedarchcopulagen("gumbel", t, n[1], Ψ[1], Φ[1]./θ)
  for i in 2:length(n)
    X = hcat(X, nestedarchcopulagen("gumbel", t, n[i], Ψ[i], Φ[i]./θ))
  end
  phi(-log.(X)./levygen(θ, rand(t)), θ, "gumbel")
end


"""
  nestedgumbelcopula(t::Int, θ::Vector{Float64})

Returns t realisations of length(θ)+1 variate data of (hierarchically) nested Gumbel copula.
C_θₙ(... C_θ₂(C_θ₁(u₁, u₂), u₃)...,  uₙ)
"""

function nestedgumbelcopula(t::Int, θ::Vector{Float64})
  issorted(θ; rev=true) || throw(AssertionError("wrong heirarchy of parameters"))
  θ = vcat(θ, [1.])
  X = rand(t,1)
  for i in 1:length(θ)-1
    X = nestedstep("gumbel", hcat(X, rand(t)), ones(t), θ[i], θ[i+1])
  end
  X
end

"""
  nestedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64})

Retenares data from nested hierarchical frechet copula
"""
nestedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α)) =
  fncopulagen(t, α, β, rand(t, length(α)+1))


function fncopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64}, u::Matrix{Float64})
  p = invperm(sortperm(u[:,end]))
  lx = floor.(Int, t.*α)
  li = floor.(Int, t.*β) + lx
  for j in size(u, 2)-1:-1:1
    u[p[1:lx[j]],j] = u[p[1:lx[j]], j+1]
    r = p[lx[j]+1:li[j]]
    u[r,j] = 1-u[r,j+1]
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
      x[:,ind] = mocopula(v, length(ind), τ2λ(ρ, λ))
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
      x[:,ind] = nestedcopulag(p[1], t, [length(s) for s in n], ϕ, θ, v)
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
  if ind[1] == "frechet"
    return i
  end
  i = vcat(i, findsimilar(transpose(x), i))
  if ind[1] == "Marshal-Olkin"
    l = length(ind[2])
    for p in 2+l:2^l-1
      i = vcat(i, findsimilar(transpose(x), i))
    end
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
