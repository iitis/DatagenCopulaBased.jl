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
 ρ2θ(ρ::Union{Float64, Int}, copula::String)

 Returns a Float, an archimedean copula parameter given expected Spermann correlation
 ρ and a copula.

"""
function ρ2θ(ρ::Union{Float64, Int}, copula::String)
  if copula == "gumbel"
    return gumbelθ(ρ)
  elseif copula == "clayton"
    return claytonθ(ρ)
  elseif copula == "frank"
    return frankθ(ρ)
  elseif copula == "amh"
    return AMHθ(ρ)
  end
  0.
end

### Helpers for given copulas

function Ccl(x::Vector{Float64}, θ::Union{Int, Float64})
  if θ > 0
    return (x[1]^(-θ)+x[2]^(-θ)-1)^(-1/θ)
  else
    return (maximum([x[1]^(-θ)+x[2]^(-θ)-1, 0]))^(-1/θ)
  end
end

Cg(x::Vector{Float64}, θ::Union{Int, Float64}) = exp(-((-log(x[1]))^θ+(-log(x[2]))^θ)^(1/θ))

gumbelρ(θ::Union{Int, Float64}) = 12*hcubature(x-> Cg(x, θ), [0,0],[1,1])[1]-3

function gumbelθ(ρ)
  function f1!(θ, fvec)
    fvec[1] = gumbelρ(θ[1])-ρ
  end
  return nlsolve(f1!, [ρ]).zero[1]
end

claytonρ(θ::Union{Int, Float64}) = 12*hcubature(x-> Ccl(x, θ), [0,0],[1,1])[1]-3

function claytonθ(ρ)
  function f1!(θ, fvec)
    fvec[1] = claytonρ(θ[1])-ρ
  end
  return nlsolve(f1!, [ρ]).zero[1]
end

dilog(x::Float64) = quadgk(t -> log(t)/(1-t), 1, x)[1]


function AMHθ(ρ::Float64)
  if ρ == 0.
    return 0.
  elseif ρ <= -0.272
    return -0.9999999
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
