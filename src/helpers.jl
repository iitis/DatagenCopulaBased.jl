lefttail(v1::Vector{T}, v2::Vector{T}, α::T = 0.002) where T <: AbstractFloat =
        sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)

 righttail(v1::Vector{T}, v2::Vector{T}, α::T = 0.998) where T <: AbstractFloat =
         sum((v1 .> α) .* (v2 .> α))./(length(v1)*(1-α))


D(θ::Float64) = 1/θ*(quadgk(i -> i/(exp(i)-1), 0, θ)[1])

function frankθ(τ::Float64)
  function f1!(θ, fvec)
    fvec[1] = 1+4*(D(θ[1])-1)/θ[1] - τ
  end
  return nlsolve(f1!, [τ]).zero[1]
end

function AMHτ2θ(τ::Float64)
  if τ >= 1/3
    return 0.999999
  elseif -2/11 < τ <1/3
    function f1!(θ, fvec)
      fvec[1] = (1 - 2*(*(1-θ[1])*(1-θ[1])log(1-θ[1]) + θ[1])/(3*θ[1]^2))-τ
    end
    return nlsolve(f1!, [τ]).zero[1]
  end
end

function ρ2θ(ρ::Union{Float64, Int}, copula::String)
  if copula == "gumbel"
    return 1/(1-2*asin(ρ)/pi)
  elseif copula == "clayton"
    return 4*asin(ρ)/(pi-2*asin(ρ))
  elseif copula == "frank"
    return 1/0.25*tan(ρ/0.7)
  elseif copula == "amh"
    return AMHθ(ρ)
  else
  return 0.
  end
end

function τ2θ(τ::Float64, copula::String)
  if copula == "gumbel"
    return 1/(1-τ)
  elseif copula == "clayton"
    return 2*τ/(1-τ)
  elseif copula == "frank"
    return frankθ(τ)
  elseif copula == "amh"
    return AMHτ2θ(τ)
  else
  return 0.
  end
end

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

using QuadGK
using NLsolve


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
0.0001*log(0.0001)


function logseriesquantile(p::Float64, v::Vector{Float64})
  w = logseriescdf(p)
  [findlast(w .< b) for b in v]
end

"""
  levygen(a::Vector{Float64}, θ)

Generates data from Levy stable distribution woth parameters α = 1/θ, β = 1,
γ = (cos(pi/(2*θ)))^θ and δ = 0
"""

function levygen(θ::Union{Int, Float64}, a::Vector{Float64})
  p = invperm(sortperm(a))
  ϕ = pi*rand(length(a))-pi/2
  v = quantile(Exponential(1.), rand(length(a)))
  γ = (cos(pi/(2*θ)))^θ
  v = ((cos.(pi/(2*θ)+(1/θ-1).*ϕ))./v).^(θ-1)
  v = γ*v.*sin.(1/θ.*(pi/2+ϕ)).*(cos(pi/(2*θ)).*cos.(ϕ)).^(-θ)
  sort(v)[p]
end


AMHτ(θ) = (1 - 2*(*(1-θ)*(1-θ)log(1-θ) + θ)/(3*θ^2))

AMHτ(-0.9999999)

2/11
