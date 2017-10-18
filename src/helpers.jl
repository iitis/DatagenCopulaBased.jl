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
  [findlast(w .< b) for b in v]
end

"""
  levygen(θ::Union{Int, Float64, u::Vector{Float64})

Pseudo cdf of Levy stable distribution with parameters α = 1/θ, β = 1,
γ = (cos(pi/(2*θ)))^θ and δ = 0.
Return Vector{Float}, given parameter ϴ and Vector{Float} - u
"""

function levygen(θ::Union{Int, Float64}, u::Vector{Float64})
  p = invperm(sortperm(u))
  ϕ = pi*rand(length(u))-pi/2
  v = quantile(Exponential(1.), rand(length(u)))
  γ = (cos(pi/(2*θ)))^θ
  v = ((cos.(pi/(2*θ)+(1/θ-1).*ϕ))./v).^(θ-1)
  v = γ*v.*sin.(1/θ.*(pi/2+ϕ)).*(cos(pi/(2*θ)).*cos.(ϕ)).^(-θ)
  sort(v)[p]
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
