# nasted copulas
# Algorithms from M. Hofert, `Efficiently sampling nested Archimedean copulas`
# Computational Statistics and Data Analysis 55 (2011) 57–70


"""
  nastedgumbelcopula(t::Int, n::Vector{Int}, Φ::Vector{Float64}, θ::Float64)

Returns t realisations of ∑ᵢ nᵢ variate data of nasted Gumbel copula
C_θ(C_Φ₁(u₁₁, ..., u₁,ₙ₁), C_θ(C_Φₖ(uₖ₁, ..., uₖ,ₙₖ)) where k = length(n).

M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
"""

function nastedgumbelcopula(t::Int, n::Vector{Int}, Φ::Vector{Float64}, θ::Float64; c::Float64 = 1.)
  θ <= minimum(Φ) || throw(AssertionError("wrong heirarchy of parameters"))
  length(n) == length(Φ) || throw(AssertionError("number of subcopulas ≠ number of parameters"))
  Φ = Φ./θ./c
  X = copulagen("gumbel", rand(t,n[1]+1), Φ[1])
  for i in 2:length(n)
    X = hcat(X, copulagen("gumbel", rand(t,n[i]+1), Φ[i]))
  end
  u = -log.(X)./levygen(θ, rand(t))
  exp.(-u.^(1/θ))
end


"""
  nastedgumbelcopulat::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, Φ::Vector{Float64}, θ₀::Float64)

Returns t realisations of ∑ᵢ ∑ⱼ nᵢⱼ variate data of double nasted Gumbel copula.
C_θ(C_Φ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_Φₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))
 where lᵢ = length(n[i])

"""


function nastedgumbelcopula(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, Φ::Vector{Float64}, θ::Float64)
  θ <= minimum(Φ) || throw(AssertionError("wrong heirarchy of parameters"))
  Φ = Φ./θ
  X = nastedgumbelcopula(t, n[1], Ψ[1], Φ[1]; c = θ)
  for i in 2:length(n)
    X = hcat(X, nastedgumbelcopula(t, n[i], Ψ[i], Φ[i], c = θ))
  end
  u = -log.(X)./levygen(θ, rand(t))
  exp.(-u.^(1/θ))
end

"""
  nastedgumbelcopula(t::Int, θ::Vector{Float64})

Returns t realisations of length(θ)+1 variate data of (hierarchically) nasted Gumbel copula.
C_θₙ(... C_θ₂(C_θ₁(u₁, u₂), u₃)...,  uₙ)
"""

function nastedgumbelcopula(t::Int, θ::Vector{Float64})
  issorted(θ; rev=true) || throw(AssertionError("wrong heirarchy of parameters"))
  copulagen("gumbel", rand(t, 2*length(θ)+1), θ)
end

function nastedclaytoncopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64)
  v = rand(t)
  V0 = quantile(Gamma(1/θ, θ), v)
  u = rand(t, n[1])
  u = -log.(u)./(V0.^(ϕ[1]) + levygen1(ϕ[1], rand(t), V0))
  X = exp.(V0.-(V0.^(θ/ϕ[1]).+u).^(ϕ[1]/θ))
  for i in 2:length(n)
    u = rand(t, n[i])
    u = -log.(u)./(V0.^(ϕ[i]) + levygen1(ϕ[i], rand(t), V0))
    X = hcat(X, exp.(V0.-(V0.^(θ/ϕ[i]).+u).^(ϕ[i]/θ)))
  end
  X
  #u = -log.(X)./V0
  #(1 + θ.*u).^(-1/θ)
end

function nastedamhcopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64)
  θ <= minimum(ϕ) || throw(AssertionError("wrong heirarchy of parameters"))
  length(n) == length(ϕ) || throw(AssertionError("number of subcopulas ≠ number of parameters"))
  v = rand(t)
  V0 = 1+quantile(Geometric(1-θ), v)
  u = rand(t, n[1])
  u = -log.(u)./(V0 + [quantile(NegativeBinomial(v, (1-ϕ[1])/(1-θ)), rand()) for v in V0])
  X = ((exp.(u)-ϕ[1])*(1-θ)+θ*(1-ϕ[1]))/(1-ϕ[1])
  for i in 2:length(n)
    u = rand(t, n[i])
    u = -log.(u)./(V0 + [quantile(NegativeBinomial(v, (1-ϕ[i])/(1-θ)), rand()) for v in V0])
    X = hcat(X, ((exp.(u)-ϕ[i])*(1-θ)+θ*(1-ϕ[i]))/(1-ϕ[i]))
  end
  for i in 1:size(X,2)
    X[:,i] = X[:,i].^(-V0)
  end
  u = -log.(X)./V0
  (1-θ)./(exp.(u)-θ)
end

function nastedfrankcopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64)
  v = rand(t)
  V0 = logseriesquantile(1-exp(-θ), v)
  u = rand(t, n[1])
  u = -log.(u)./frankgen(ϕ[1], θ, V0)
  X = (1-(1-exp.(-u)*(1-exp(-ϕ[1]))).^(θ/ϕ[1]))./(1-exp(-θ))
  for i in 2:length(n)
    u = rand(t, n[i])
    u = -log.(u)./frankgen(ϕ[i], θ, V0)
    X = hcat(X,(1-(1-exp.(-u)*(1-exp(-ϕ[i]))).^(θ/ϕ[i]))./(1-exp(-θ)))
  end
  u = -log.(X)
  -log.(1+exp.(-u)*(exp(-θ)-1))/θ
end

"""
  copulagen(copula::String, r::Matrix{Float}, θ::Vector{Float64})

Auxiliary function used to generate data from nasted (hiererchical) gumbel copula
parametrised by a single parameter θ given a matrix of independent [0,1] distributerd
random vectors.

"""

function copulagen(copula::String, r::Matrix{T}, θ::Vector{Float64}) where T <:AbstractFloat
  if copula == "gumbel"
    θ = vcat(θ, [1.])
    n = length(θ)
    u = r[:,1:n]
    v = r[:,n+1:end]
    X = copulagen(copula, hcat(u[:,1:2], v[:,1]), θ[1]/θ[2])
    for i in 2:(n-1)
      X = hcat(X, u[:,i+1])
      X = -log.(X)./levygen(θ[i]/θ[i+1], v[:,i])
      X = exp.(-X.^(θ[i+1]/θ[i]))
    end
    return X
  end
  r
end

"""
  nastedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64})

Retenares data from nasted hierarchical frechet copula
"""
function nastedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))
  α = vcat([0.], α)
  β = vcat([0.], β)
  n = length(α)
  u = rand(t, n)
  p = invperm(sortperm(u[:,1]))
  l = floor.(Int, t.*α)
  lb = floor.(Int, t.*β)
  for i in 2:n
    u[1:l[i],i] = u[1:l[i], i-1]
    r = l[i]+1:lb[i]+l[i]
    u[r,i] = 1-u[r,i-1]
  end
  u[p,:]
end

# copula mix

function copulamix(t::Int, Σ::Matrix{Float64}, inds::Vector{Pair{String,Vector{Int64}}},
                                                λ::Vector{Float64} = [0.8, 0.1],
                                                ν::Int = 10)
  x = transpose(rand(MvNormal(Σ),t))
  xgauss = copy(x)
  x = cdf(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = norm2unifind(xgauss, Σ, makeind(Σ, p))
    if p[1] == "Marshal-Olkin"
      map = collect(combinations(1:length(ind),2))
      ρ = [Σ[ind[k[1]], ind[k[2]]] for k in map]
      x[:,ind] = mocopula(v, length(ind), τ2λ(ρ, λ))
    elseif (p[1] == "gumbel") & (length(ind) > 2)
      θ = [ρ2θ(Σ[ind[i], ind[i+1]], p[1]) for i in 1:(length(ind)-1)]
      x[:,ind] = copulagen(p[1], v, sort(θ; rev = true))
    elseif p[1] == "t-student"
      g2tsubcopula!(x, Σ, ind, ν)
    else
      θ = ρ2θ(Σ[ind[1], ind[2]], p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    end
  end
  x
end

"""
  makeind(Σ::Matrix{Float64}, ind::Pair{String,Vector{Int64}})

Returns multiindex hcat(ind[2], [j₁, ..., jₖ]) where js are such that maximise Σ[ind[2] [i], js]
k is determined by the copula type ind[1] and length(ind[2])
"""

function makeind(Σ::Matrix{Float64}, ind::Pair{String,Vector{Int64}})
  l = length(ind[2])
  i = ind[2]
  lim = l+1
  if ind[1] =="Marshal-Olkin"
    lim = 2^l-1
  elseif ind[1] =="gumbel"
    lim = 2*l-1
  end
  for p in 0:(lim-l-1)
    k = p%l+1
    i = vcat(i, find(Σ[:, k].== maximum(Σ[setdiff(collect(1:size(Σ, 2)),i),k])))
  end
  i
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
