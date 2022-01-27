"""
    logseriescdf(p)

Returns the vector of samples from the discrete cdf of logarithmic distribution
"""
function logseriescdf(p)
  T = eltype(p)
  cdfs = T.([0.])
  for i in 1:100000000
    @inbounds push!(cdfs, cdfs[i]-(p^i)/(i*log(1-p)))
    # one can change atol here
    if isapprox(cdfs[i],  1.)
      return cdfs
    end
  end
  cdfs
end

"""
    levyel(θ, u1, u2)

u1 and u2 are supposed to be random numbers form [0,1]
An element from Levy stable distribution with parameters α = 1/θ, β = 1,
γ = (cos(pi/(2*θ)))^θ and δ = 0.
"""
function levyel(θ, u1, u2)
  ϕ = pi*u1-pi/2
  v = quantile.(Exponential(1.), u2)
  γ = (cos(pi/(2*θ)))^θ
  v = ((cos(pi/(2*θ)+(1/θ-1)*ϕ))/v)^(θ-1)
  γ*v*sin(1/θ*(pi/2+ϕ))*(cos(pi/(2*θ))*cos(ϕ))^(-θ)
end

"""
    tiltedlevygen(V0, α; rng::AbstractRNG)

Returns a sample from the expotencialy tilted Levy stable pdf
f(x; V0, α) = exp(-V0^α) g(x; α)/exp(-V0), where g(x; α) is a stable Levy pdf
with parameters α = 1/θ, β = 1, γ = (cos(pi/(2*θ)))^θ and δ = 0.
"""
function tiltedlevygen(V0, α; rng::AbstractRNG)
  T = eltype(V0)
  x = levyel(α, rand(rng, T), rand(rng, T))
  u = rand(rng, T)
  while exp(-V0^α*x)/(1500*exp(-V0)) < u
    x = levyel(α, rand(rng, T), rand(rng, T))
    u = rand(rng, T)
  end
  return x.*V0.^α
end

"""
    Ginv(y, α)

Returns Real, helper for the joe/frank nested copula generator
"""
Ginv(y, α) = ((1-y)*SpecialFunctions.gamma(1-α))^(-1/α)

"""
    InvlaJ(n::Int, α)

Returns the inverse Laplace transform of generator of Joe nested copula
parametered by α
"""
function InvlaJ(n::Int, α)
  T = eltype(α)
  return 1-1/(n*beta(T(n), 1-α))
end

"""
  sampleInvlaJ(α, v)

Returns Int, a sample of inverce laplacea transform of generator of Joe nested copula,
given a parameter α and a random numver v ∈ [0,1]
"""
function sampleInvlaJ(α, v)
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
  elInvlaF(θ₁, θ₀, logseriescdf; rng::AbstractRNG)

Returns Int, a single sample of the inverse Laplace transform of the generator
of nested Frank copula
"""
function elInvlaF(θ₁, θ₀, logseriescdf; rng::AbstractRNG)
  T = eltype(θ₁)
  c1 = 1-exp(-θ₁)
  α = θ₀/θ₁
  if θ₀ <= 1
    v = rand(rng, T)
    X = findlast(logseriescdf .< rand(rng, T))[1]
    while v > 1/((X-α)*beta(X, 1-α))
      v = rand(rng, T)
      X = findlast(logseriescdf .< rand(rng, T))[1]
    end
    return X
  else
    v = rand(rng, T)
    X = sampleInvlaJ(α, rand(rng, T))
    while v > c1^(X-1)
      X = sampleInvlaJ(α, rand(rng, T))
      v = rand(rng, T)
    end
    return X
  end
end

"""
  nestedfrankgen(θ₁, θ₀, V0::Int, logseriescdf; rng::AbstractRNG)

Return int, sampled from the Inverse Laplace trensform of nested
Frank copula given parametes θ₁ θ₀ (child and parent)
and an element of V0 - vector of samples of invlaplace of the parents copula
"""
function nestedfrankgen(θ₁, θ₀, V0::Int, logseriescdf; rng::AbstractRNG)
  T = eltype(θ₁)
  return sum([elInvlaF(θ₁, θ₀, logseriescdf; rng = rng) for j in 1:V0])
end
