
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
  f(τ) = 1/2.*sin.(τ[1]*pi/2)+τ[1]/2 - ρ[1]
  fzero(f, -0.999, 0.999)
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
