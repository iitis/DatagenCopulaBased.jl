# nested copulas
# Algorithms from M. Hofert, `Efficiently sampling nested Archimedean copulas`
# Computational Statistics and Data Analysis 55 (2011) 57–70


"""
  nestedgumbelcopula(t::Int, n::Vector{Int}, Φ::Vector{Float64}, θ::Float64)

Returns t realisations of ∑ᵢ nᵢ variate data of nested Gumbel copula
C_θ(C_Φ₁(u₁₁, ..., u₁,ₙ₁), C_θ(C_Φₖ(uₖ₁, ..., uₖ,ₙₖ)) where k = length(n).

M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
"""

function nestedgumbelcopula(t::Int, n::Vector{Int}, Φ::Vector{Float64}, θ::Float64, m::Int = 0;
                                                                                    c::Float64 = 1.)
  testnestedpars(θ, Φ, n)
  Φ = Φ./θ./c
  X = copulagen("gumbel", rand(t,n[1]+1), Φ[1])
  for i in 2:length(n)
    X = hcat(X, copulagen("gumbel", rand(t,n[i]+1), Φ[i]))
  end
  if m != 0
    X = hcat(X, rand(t,m))
  end
  u = -log.(X)./levygen(θ, rand(t))
  exp.(-u.^(1/θ))
end


"""
  nestedgumbelcopulat::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, Φ::Vector{Float64}, θ₀::Float64)

Returns t realisations of ∑ᵢ ∑ⱼ nᵢⱼ variate data of double nested Gumbel copula.
C_θ(C_Φ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_Φₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))
 where lᵢ = length(n[i])

"""


function nestedgumbelcopula(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, Φ::Vector{Float64}, θ::Float64)
  θ <= minimum(Φ) || throw(AssertionError("wrong heirarchy of parameters"))
  Φ = Φ./θ
  X = nestedgumbelcopula(t, n[1], Ψ[1], Φ[1]; c = θ)
  for i in 2:length(n)
    X = hcat(X, nestedgumbelcopula(t, n[i], Ψ[i], Φ[i], c = θ))
  end
  u = -log.(X)./levygen(θ, rand(t))
  exp.(-u.^(1/θ))
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

function nestedstep(copula::String, u::Matrix{Float64}, v::Vector{Float64},
                                                        V0::Union{Vector{Float64}, Vector{Int}},
                                                        ϕ::Float64, θ::Float64)
  if copula == "amh"
    t = length(V0)
    w = [quantile(NegativeBinomial(V0[i], (1-ϕ)/(1-θ)), v[i]) for i in 1:t]
    u = -log.(u)./(V0 + w)
    return ((exp.(u)-ϕ)*(1-θ)+θ*(1-ϕ))/(1-ϕ)
  elseif copula == "frank"
    u = -log.(u)./frankgen(ϕ, θ, V0)
    return (1-(1-exp.(-u)*(1-exp(-ϕ))).^(θ/ϕ))./(1-exp(-θ))
  elseif copula == "clayton"
    #u = -log.(u)./gens(V0, θ/ϕ)
    u = -log.(u)./ge(V0, ϕ/θ)
    return exp.(V0.-V0.*(1.+u).^(θ/ϕ))
  end
  u
end

function nestedclaytoncopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  testnestedpars(θ, ϕ, n)
  v = rand(t)
  V0 = quantile(Gamma(1/θ, θ), v)
  X = nestedstep("clayton", rand(t, n[1]), rand(t), V0, ϕ[1], θ)
  for i in 2:length(n)
    X = hcat(X, nestedstep("clayton", rand(t, n[i]), rand(t), V0, ϕ[i], θ))
  end
  if m != 0
    X = hcat(X, rand(t, m))
  end
  u = -log.(X)./V0
  (1 + θ.*u).^(-1/θ)
end

"""
  nestedamhcopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested Ali-Mikhail-Haq copula, outer copula parameter is θ, inner i'th copulas parameter is
ϕ[i] and size is n[i]. If m ≠ 0, last m variables are from outer copula only, see Alg. 5
McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'. Journal of Statistical Computation and Simulation 78, 567–581.
"""

function nestedamhcopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  testnestedpars(θ, ϕ, n)
  v = rand(t)
  V0 = 1+quantile(Geometric(1-θ), v)
  X = nestedstep("amh", rand(t, n[1]), rand(t), V0, ϕ[1], θ)
  for i in 2:length(n)
    X = hcat(X, nestedstep("amh", rand(t, n[i]), rand(t), V0, ϕ[i], θ))
  end
  if m != 0
    X = hcat(X, 1./rand(t, m).^(1./V0))
  end
 (1-θ)./(X-θ)
end

"""
  nestedfrankcopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Returns Matrix{Float} of t realisations of sum(n)+m random variables generated using
nested Ali-Mikhail-Haq copula, outer copula parameter is θ, inner i'th copulas parameter is
ϕ[i] and size is n[i]. If m ≠ 0, last m variables are from outer copula only
"""

function nestedfrankcopula(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  testnestedpars(θ, ϕ, n)
  v = rand(t)
  V0 = logseriesquantile(1-exp(-θ), v)
  X = nestedstep("frank", rand(t, n[1]), rand(t), V0, ϕ[1], θ)
  for i in 2:length(n)
    X = hcat(X, nestedstep("frank", rand(t, n[i]), rand(t), V0, ϕ[i], θ))
  end
  if m != 0
    X = hcat(X, rand(t, m).^(1./V0))
  end
  -log.(1+X*(exp(-θ)-1))/θ
end

"""
  nestedgumbelcopula(t::Int, θ::Vector{Float64})

Returns t realisations of length(θ)+1 variate data of (hierarchically) nested Gumbel copula.
C_θₙ(... C_θ₂(C_θ₁(u₁, u₂), u₃)...,  uₙ)
"""

function nestedgumbelcopula(t::Int, θ::Vector{Float64})
  issorted(θ; rev=true) || throw(AssertionError("wrong heirarchy of parameters"))
  hiergcopulagen(rand(t, 2*length(θ)+1), θ)
end

"""
  hiergcopulagen(r::Matrix{Float}, θ::Vector{Float64})

Auxiliary function used to generate data from nested (hiererchical) gumbel copula
parametrised by a single parameter θ given a matrix of independent [0,1] distributerd
random vectors.

"""

function hiergcopulagen(r::Matrix{T}, θ::Vector{Float64}) where T <:AbstractFloat
  n = length(θ)+1
  u = r[:,1:n]
  v = r[:,n+1:end]
  θ = vcat(θ, [1.])
  X = copulagen("gumbel", hcat(u[:,1:2], v[:,1]), θ[1]/θ[2])
  for i in 2:(n-1)
    X = hcat(X, u[:,i+1])
    X = -log.(X)./levygen(θ[i]/θ[i+1], v[:,i])
    X = exp.(-X.^(θ[i+1]/θ[i]))
  end
  X
end

"""
  nestedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64})

Retenares data from nested hierarchical frechet copula
"""
function nestedfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))
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
      x[:,ind] = hiergcopulagen(v, sort(θ; rev = true))
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
