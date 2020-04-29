"""
    logseriescdf(p::Float64)

Returns the vector of samples of the discrete cdf of logarithmic distribution
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
    levyel(θ::Float64, u1::Float64, u2::Float64)

u1 and u2 are supposed to be random numbers form [0,1]
An element from Levy stable distribution with parameters α = 1/θ, β = 1,
γ = (cos(pi/(2*θ)))^θ and δ = 0.
Return Float, given parameter ϴ of dostribution
"""
function levyel(θ::Float64, u1::Float64, u2::Float64)
  ϕ = pi*u1-pi/2
  v = quantile.(Exponential(1.), u2)
  γ = (cos(pi/(2*θ)))^θ
  v = ((cos(pi/(2*θ)+(1/θ-1)*ϕ))/v)^(θ-1)
  γ*v*sin(1/θ*(pi/2+ϕ))*(cos(pi/(2*θ))*cos(ϕ))^(-θ)
end

"""
    tiltedlevygen(V0::Float64, α::Float64; rng::AbstractRNG)

Returns a Float64 sampled  from the expotencialy tilted Levy stable pdf
f(x; V0, α) = exp(-V0^α) g(x; α)/exp(-V0), where g(x; α) is a stable Levy pdf
with parameters α = 1/θ, β = 1, γ = (cos(pi/(2*θ)))^θ and δ = 0.
"""
function tiltedlevygen(V0::Float64, α::Float64; rng::AbstractRNG)
  x = levyel(α, rand(rng), rand(rng))
  u = rand(rng)
  while exp(-V0^α*x)/(1500*exp(-V0)) < u
    x = levyel(α, rand(rng), rand(rng))
    u = rand(rng)
  end
  return x.*V0.^α
end

"""
    Ginv(y::Float64, α::Float64)

Returns Float64, helper for the joe/frank nested copula generator
"""
Ginv(y::Float64, α::Float64) = ((1-y)*SpecialFunctions.gamma(1-α))^(-1/α)

"""
    InvlaJ(n::Int, α::Float64)

Returns Float64 from the inverse laplacea transform of generator of Joe nested copula
parametered by α
"""
InvlaJ(n::Int, α::Float64) = 1-1/(n*beta(1.0*n, 1-α))

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
  elInvlaF(θ₁::Float64, θ₀::Float64, logseriescdf::Vector{Float64}; rng::AbstractRNG)

Returns Int, a single sample of the inverse laplacea transform of the generator
of nested Frank copula
"""
function elInvlaF(θ₁::Float64, θ₀::Float64, logseriescdf::Vector{Float64}; rng::AbstractRNG)
  c1 = 1-exp(-θ₁)
  α = θ₀/θ₁
  if θ₀ <= 1
    v = rand(rng)
    X = findlast(logseriescdf .< rand(rng))[1]
    while v > 1/((X-α)*beta(X, 1-α))
      v = rand(rng)
      X = findlast(logseriescdf .< rand(rng))[1]
    end
    return X
  else
    v = rand(rng)
    X = sampleInvlaJ(α, rand(rng))
    while v > c1^(X-1)
      X = sampleInvlaJ(α, rand(rng))
      v = rand(rng)
    end
    return X
  end
end

"""
  nestedfrankgen(θ₁::Float64, θ₀::Float64, V0::Int, logseriescdf::Vector{Float64}; rng::AbstractRNG))

Return int, sampled from the Inverse Laplace trensform of nested
Frank copula given parametes θ₁ θ₀ (child and parent)
and an element of V0 - vector of samples of invlaplace of the parents copula
"""
function nestedfrankgen(θ₁::Float64, θ₀::Float64, V0::Int, logseriescdf::Vector{Float64}; rng::AbstractRNG)
    return sum([elInvlaF(θ₁, θ₀, logseriescdf; rng = rng) for j in 1:V0])
end
