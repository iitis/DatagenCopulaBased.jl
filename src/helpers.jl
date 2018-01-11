
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
    ret = map(x -> (x > 0)? (x): (0.000000001), inv(M)*fm*λ)
    [λ[1:3]..., ret..., λ[end]]
  end
end

function moρ2τ(ρ::Float64)
  function f1!(τ, fvec)
    fvec[1] = 1/2.*sin.(τ[1]*pi/2)+τ[1]/2 - ρ[1]
  end
  return nlsolve(f1!, [ρ]).zero[1]
end

"""
  frechetρ2αβ(ρ::Vector{Float64}, a::Vector{Float64})

Returns Vector{Float}, Vector{Float}, parameters of the frechet copula given a
sequential correlation and parameter a = α- β or β - α
"""
function frechetρ2αβ(ρ::Vector{Float64}, a::Vector{Float64})
  la = length(a)
  l = length(ρ)
  a = (l > la)? hcat(a, transpose(0.1*ones(l - la))): a
  a = [minimum([a[i], 0.5-ρ[i]/2, 0.5+ρ[i]/2]) for i in 1:l]
  α = [(ρ[i] >= 0)? ρ[i] + a[i]: a[i] for i in 1:l]
  β = [(ρ[i] < 0)? -ρ[i] + a[i]: a[i] for i in 1:l]
  α, β
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
