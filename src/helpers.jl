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
  levygen(θ::Union{Int, Float64, u::Vector{Float64})

Pseudo cdf of Levy stable distribution with parameters α = 1/θ, β = 1,
γ = (cos(pi/(2*θ)))^θ and δ = 0.
Return Vector{Float}, given parameter ϴ and Vector{Float} - u
"""

function levyg(θ::Union{Int, Float64}, t::Int)
  ϕ = pi*rand(t)-pi/2
  v = quantile(Exponential(1.), rand(t))
  γ = (cos(pi/(2*θ)))^θ
  v = ((cos.(pi/(2*θ)+(1/θ-1).*ϕ))./v).^(θ-1)
  γ*v.*sin.(1/θ.*(pi/2+ϕ)).*(cos(pi/(2*θ)).*cos.(ϕ)).^(-θ)
end



#x = levyg(1.5, 1000000)
#u = ecdf(x);
#u(2)

function ge(V0::Vector{Float64}, α::Float64)
  t = length(V0)
  ret = zeros(t)
  for i in 1:t
    x = levyg(α, 1)[1]
    u = rand()
    while exp(-V0[i]^α*x)/(10*exp(-V0[i])) < u
      x = levyg(α, 1)[1]
      u = rand()
    end
    ret[i] = x
  end
  ret.*V0.^α
end

#=
using Distributions
using StatsBase
r = 0.1*rand(500000)
x = ge(r, 4.)
y = gens(r, 1/4.)

mean(x)
std(x)
skewness(x)
kurtosis(x)

mean(y)
std(y)
skewness(y)
kurtosis(y)

exp(0.2^1/5)
=#



function levygen(θ::Union{Int, Float64}, u::Vector{Float64})
  p = invperm(sortperm(u))
  v = levyg(θ, length(u))
  sort(v)[p]
end


function rgen(V0::Float64, α::Float64, γ::Float64)
  w = quantile(Exponential(1.), rand())
  minimum([(V0/(gamma(1-α)*γ))^(1/α), w*(rand())^(1/α)])
end


function el(v::Float64, α::Float64)
  γ = quantile(Exponential(1.), rand())
  ret = 0.
  for k in 1:floor(Int, 100000/α^4)
    @inbounds temp = rgen(v, α, γ)
    γ += quantile(Exponential(1.), rand())
    ret += temp
    if temp < 1.e-12
      return ret
    end
  end
  ret
end

function gens(V0::Vector{Float64}, α::Float64)
  if nprocs() == 1
    ret = zeros(V0)
    for i in 1:length(V0)
      @inbounds ret[i] = el(V0[i], α)
    end
    return ret
  else
    return pmap(v -> el(v, α), V0)
  end
end


"""
  Ginv(y::Float64, α::Float64)

Returns Float64, helper for the joe/frank nested copula generator
"""
Ginv(y::Float64, α::Float64) = ((1-y)*gamma(1-α))^(-1/α)

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
      return (InvlaJ(floor(Int, G), α) < v)? ceil(Int, G): floor(Int, G)
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
  frankngen(θ₁::Float64, θ₀::Float64, V0::Vector{Int})

Return vector of int, samples of inverse laplacea trensform of nested
Frak copula given parametes and V0 - vector of samples if invlaplace of perents copula
"""

function frankgen(θ₁::Float64, θ₀::Float64, V0::Vector{Int})
  if nprocs() == 1
    return map(k -> sum([elInvlaF(θ₁, θ₀) for j in 1:k]), V0)
  end
  u = SharedArray{Float64}(length(V0))
  @sync @parallel for i = 1:length(V0)
    u[i] = sum([elInvlaF(θ₁, θ₀) for j in 1:V0[i]])
  end
  Array(u)
end


"""
  function tail(v1::Vector{Float}, v2::Vector{Float}, α::Float = 0.002, tail::String)


Returns empirical left and right tail dependency of bivariate data
"""

function tail(v1::Vector{T}, v2::Vector{T}, tail::String, α::T = 0.002) where T <: AbstractFloat
  if tail == "l"
    return sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)
  elseif tail == "r"
    return sum((v1 .> (1-α)) .* (v2 .> (1-α)))./(length(v1)*α)
  end
  0.
end

"""
  Debye(x::Float64, k::Int)

Returns float64, Debye function Dₖ(x) value
"""

Debye(x::Float64, k::Int=1) = k/x^k*(quadgk(i -> i^k/(exp(i)-1), 0, x)[1])

# kendall's τ to copulas parameters

"""
  τ2θ(τ::Float64, copula::String)

Returns Float, a single parameter of Archimedean copula, given the Kenalss τ correlation
"""
function τ2θ(τ::Float64, copula::String)
  if copula == "gumbel"
    return 1/(1-τ)
  elseif copula == "clayton"
    return 2*τ/(1-τ)
  elseif copula == "frank"
    return frankτ2θ(τ)
  elseif copula == "amh"
    return AMHτ2θ(τ)
  else
  return 0.
  end
end
"""
  τ2λ(τ::Vector{Float64}, λ::Vector{Float64})

Suplement the vector of λ patrameters of Marshal-Olkin copula, given some of those
parameters and a vector fo Kendall's τ correlations. Wroks fo 2,3 variate MO copula
"""

function τ2λ(τ::Vector{Float64}, λ::Vector{Float64})
  if length(τ) == 1
    return [λ[1],λ[2],(λ[1]+λ[2])*τ[1]/(1-τ[1])]
  else
    t = τ./(1-τ)
    M = eye(3) -(ones(3,3)-eye(3)) .*t
    fm = hcat([1. 1. 0.; 0. 1. 1.; 1. 0. 1.].*t, -[1.; 1.; 1.])
    ret = map(x -> (x > 0)? (x): (0.01), inv(M)*fm*λ)
    [λ[1:3]..., ret..., λ[end]]
  end
end

"""
  frankτ2θ(τ::Float64)

Returns a Frank copula θ parameter, givem Kendall's τ
"""

function frankτ2θ(τ::Float64)
  function f1!(θ, fvec)
    fvec[1] = 1+4*(Debye(θ[1])-1)/θ[1] - τ
  end
  return nlsolve(f1!, [τ]).zero[1]
end

"""
  AMHτ2θ(τ::Float64)

Returns Ali-Mikhail-Haq copula θ parameter, givem Kendall's τ
"""


function AMHτ2θ(τ::Float64)
  if τ >= 0.28
    return 0.999999
  elseif -2/11 < τ < 0.28
    function f1!(θ, fvec)
      fvec[1] = (1 - 2*(*(1-θ[1])*(1-θ[1])log(1-θ[1]) + θ[1])/(3*θ[1]^2))-τ
    end
    return nlsolve(f1!, [τ]).zero[1]
  end
  -0.999999999
end

# pearson ρ to copulas parameter

"""
 ρ2τ(ρ::Float64)

 Returns a Float, a clayton or gumbel copula kendall's tau correlation, given
pearson correlation, uses the empirical model

"""
ρ2τ(ρ::Float64) = (ρ < 0.75)?  0.622*ρ +0.157*ρ^2: -0.04+0.207*ρ-0.075*ρ^2+0.579*asin(ρ)

function ρ2θ(ρ::Union{Float64, Int}, copula::String)
  if copula == "gumbel"
    return 1/(1-ρ2τ(ρ))
  elseif copula == "clayton"
    τ = (ρ < 0)? -ρ2τ(-ρ): ρ2τ(ρ)
    return 2*τ/(1-τ)
  elseif copula == "frank"
    return frankθ(ρ)
  elseif copula == "amh"
    return AMHθ(ρ)
  end
  0.
end

dilog(x::Float64) = quadgk(t -> log(t)/(1-t), 1, x)[1]


function AMHθ(ρ::Float64)
  if ρ == 0.
    return 0.
  elseif -0.272 < ρ < 0.475
    function f1!(θ, fvec)
      p = θ[1]
      fvec[1] = (12*(1+p)*dilog(1-p)-24*(1-p)*log(1-p))/p^2-3*(p+12)/p-ρ
    end
    return nlsolve(f1!, [ρ]).zero[1]
  end
  0.99999999
end

function frankθ(ρ::Float64)
  function f1!(θ, fvec)
    fvec[1] = 1+12*(Debye(θ[1], 2)- Debye(θ[1]))/θ[1]-ρ
  end
  return nlsolve(f1!, [ρ]).zero[1]
end

# some Marshal-Olkin copulas helpers

function getmoλ(λ::Vector{Float64}, ind::Vector{Int})
  n = floor(Int, log(2, length(λ)+1))
  s = collect(combinations(1:n))
  λ[[ind == a for a in s]]
end

function setmoλ!(λ::Vector{Float64}, ind::Vector{Int}, a::Float64)
  n = floor(Int, log(2, length(λ)+1))
  s = collect(combinations(1:n))
  λ[[ind == a for a in s]] = a
end


#using Combinatorics
#ind(k::Int, s::Vector{Vector{Int}}) = [k in s[i] for i in 1:size(s,1)]
#s = collect(combinations(1:3))
#s[[length(s[i]) == 2 for i in 1:size(s,1)]]
#τ2λ([0.1, 0.5, 0.8], [0.1, 0.2, .3, 1.0])

#using QuadGK
#using NLsolve

#=
using Distributions

θ = 3.
ϕ = 4
V0 = quantile(Gamma(1/θ, θ), 0.1)


[exp.(V0.-(V0.^(θ/ϕ[1])+u).^(ϕ[1]/θ)) for u in 0:0.1:100]

=#
