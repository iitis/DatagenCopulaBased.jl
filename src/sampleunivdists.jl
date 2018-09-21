"""
  logseriescdf(p::Float64)

Returns a vector{Float} of the discrete cdf of logarithmic distribution
"""
function logseriescdf(p::Float64)
  cdfs = [0.]
  for i in 1:100000000
    @inbounds push!(cdfs, cdfs[i]-(p^i)/(i*log(1-p)))
    if cdfs[i] ≈ 1.0
      return cdfs
    end
  end
  cdfs
end

"""
  logseriesquantile(p::Float64, v::Vector{Float64})

Returns a vector{Float} of the v[i] th quaintlie of logaritmic distribution with parameter p
"""
function logseriesquantile(p::Float64, v::Vector{Float64})
  w = logseriescdf(p)
  pmap(i -> findlast(w .< i), v)
end

"""
  levyel(θ::Union{Int, Float64})

An element from Levy stable distribution with parameters α = 1/θ, β = 1,
γ = (cos(pi/(2*θ)))^θ and δ = 0.
Return Float, given parameter ϴ of dostribution
"""
function levyel(θ::Union{Int, Float64})
  ϕ = pi*rand()-pi/2
  v = quantile.(Exponential(1.), rand())
  γ = (cos(pi/(2*θ)))^θ
  v = ((cos(pi/(2*θ)+(1/θ-1)*ϕ))/v)^(θ-1)
  γ*v*sin(1/θ*(pi/2+ϕ))*(cos(pi/(2*θ))*cos(ϕ))^(-θ)
end

"""
Return a Vector{Float64} of  of pseudo cdf of Levy stable distribution with parameters
α = 1/θ, β = 1, γ = (cos(pi/(2*θ)))^θ and δ = 0, given a vector of Float64 - u

"""
function levygen(θ::Union{Int, Float64}, u::Vector{Float64})
  p = invperm(sortperm(u))
  v = [levyel(θ) for a in u]
  sort(v)[p]
end

"""
  tiltedlevygen(V0::Vector{Float64}, α::Float64)

Returns a Vector{Float} genrated from the expotencialy tilted levy stable pdf
f(x; V0, α) = exp(-V0^α) g(x; α)/exp(-V0), where g(x; α) is a stable Levy pdf
with parameters α = 1/θ, β = 1, γ = (cos(pi/(2*θ)))^θ and δ = 0.

"""
function tiltedlevygen(V0::Vector{Float64}, α::Float64)
  t = length(V0)
  ret = zeros(t)
  for i in 1:t
    x = levyel(α)
    u = rand()
    while exp(-V0[i]^α*x)/(1500*exp(-V0[i])) < u
      x = levyel(α)
      u = rand()
    end
    ret[i] = x
  end
  ret.*V0.^α
end

"""
  Ginv(y::Float64, α::Float64)

Returns Float64, helper for the joe/frank nested copula generator
"""
Ginv(y::Float64, α::Float64) = ((1-y)*SpecialFunctions.gamma(1-α))^(-1/α)

"""
  InvlaJ(n::Int, α::Float64)

Returns Float64, n-th element of the inverse laplacea transform of generator of Joe nested copula
"""
InvlaJ(n::Int, α::Float64) = 1-1/(n*beta(n, 1-α))

"""
  sampleInvlaJ(α::Float64, v::Float64)

Returns Int, a sample of inverce laplacea transform of generator of Joe nested copula,
given a parameter α and a random numver v ∈ [0,1]
"""
function sampleInvlaJ(α::Float64, v::Float64)
  if v <= α
    return 1
  else
    G = Ginv(v, α)
    if G > 2^62
      return 2^62
    else
      return (InvlaJ(floor(Int, G), α) < v) ? ceil(Int, G) : floor(Int, G)
    end
  end
end

"""
  elInvlaF(θ₁::Float64, θ₀::Float64)

Returns Int, a single sample of the inverse laplacea transform of the generator
of nested Frank copula
"""
function elInvlaF(θ₁::Float64, θ₀::Float64)
  c1 = 1-exp(-θ₁)
  α = θ₀/θ₁
  if θ₀ <= 1
    v = rand()
    X = logseriesquantile(c1, rand(1))[1]
    while v > 1/((X-α)*beta(X, 1-α))
      v = rand()
      X = logseriesquantile(c1, rand(1))[1]
    end
    return X
  else
    v = rand()
    X = sampleInvlaJ(α, rand())
    while v > c1^(X-1)
      X = sampleInvlaJ(α, rand())
      v = rand()
    end
    return X
  end
end

"""
  nestedfrankgen(θ₁::Float64, θ₀::Float64, V0::Vector{Int})

Return vector of int, samples of inverse laplacea trensform of nested
Frank copula given parametes and V0 - vector of samples if invlaplace of perents copula
"""
function nestedfrankgen(θ₁::Float64, θ₀::Float64, V0::Vector{Int})
  if nprocs() == 1
    return map(k -> sum([elInvlaF(θ₁, θ₀) for j in 1:k]), V0)
  end
  u = SharedArray{Float64}(length(V0))
  @sync @distributed for i = 1:length(V0)
    u[i] = sum([elInvlaF(θ₁, θ₀) for j in 1:V0[i]])
  end
  Array{Int}(u)
end
