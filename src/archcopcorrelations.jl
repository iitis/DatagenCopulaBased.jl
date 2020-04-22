"""
  Debye(x::Float64, k::Int)

Returns float64, Debye function Dₖ(x) value
"""
Debye(x, k::Int=1) = k/x^k*(quadgk(i -> i^k/(exp(i)-1), 0, x)[1])

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
  throw(AssertionError("$(copula) not supported"))
  end
end

"""
  frankτ2θ(τ::Float64)

Returns a Frank copula θ parameter, givem Kendall's τ
"""
function frankτ2θ(τ::Float64)
  f(θ) = 1+4*(Debye(θ)-1)/θ - τ
  if τ > 0.
    return fzero(f, τ, 100.)
  elseif τ < 0.
    return fzero(f, -100., τ)
  end
  throw(DomainError("τ = 0 not supported"))
end

"""
  AMHτ2θ(τ::Float64)

Returns Ali-Mikhail-Haq copula θ parameter, givem Kendall's τ
"""
function AMHτ2θ(τ::Float64)
  f(θ) = (1 - 2*(*(1-θ)*(1-θ)log(1-θ) + θ)/(3*θ^2))-τ
  if -0.01 < τ < 0.01
    return 0.0000000000001
  elseif τ >= 0.28
    return 0.9999
  elseif 0. < τ < 0.28
    return fzero(f, 0.005, 0.995)
  elseif -2/11 < τ < 0.
    return fzero(f, -0.995, -0.005)
  end
  -0.9999
end

# Spearman ρ to copulas parameter

"""
 ρ2θ(ρ::Union{Float64, Int}, copula::String)

 Returns a Float, an archimedean copula parameter given expected Spermann correlation
 ρ and a copula.

"""
function ρ2θ(ρ::Union{Float64, Int}, copula::String)
  if copula == "gumbel"
    return gumbelρ2θ(ρ)
  elseif copula == "clayton"
    return claytonρ2θ(ρ)
  elseif copula == "frank"
    return frankρ2θ(ρ)
  elseif copula == "amh"
    return AMHρ2θ(ρ)
  end
  throw(AssertionError("$(copula) not supported"))
end

### Clayton and gumbel copulas

function Ccl(x, θ::Union{Int, Float64})
  if θ > 0
    return (x[1]^(-θ)+x[2]^(-θ)-1)^(-1/θ)
  else
    return (maximum([x[1]^(-θ)+x[2]^(-θ)-1, 0]))^(-1/θ)
  end
end

Cg(x, θ::Union{Int, Float64}) = exp(-((-log(x[1]))^θ+(-log(x[2]))^θ)^(1/θ))

dilog(x) = quadgk(t -> log(t)/(1-t), 1, x)[1]

# converts parameter to correlations and vice versa

gumbelθ2ρ(θ) = 12*hcubature(x-> Cg(x, θ), [0,0],[1,1])[1]-3

function gumbelρ2θ(ρ)
  if ρ < 0.01
    return 1.
  else
    f(θ) = gumbelθ2ρ(θ)-ρ
    return fzero(f, 1.000001, 100.)
  end
end

 claytonθ2ρ(θ) = 12*hcubature(x-> Ccl(x, θ), [0,0],[1,1])[1]-3

function claytonρ2θ(ρ)
  f(θ) = claytonθ2ρ(θ)-ρ
  if ρ > .038
    return fzero(f, .001, 100.)
  elseif ρ < -.038
    return fzero(f, -1., -0.001)
  else
    return 0.052
  end
end

function AMHρ2θ(ρ::Float64)
  f(p) = (12*(1+p)*dilog(1-p)-24*(1-p)*log(1-p))/p^2-3*(p+12)/p-ρ
  if -0.01 < ρ  < 0.01
    return 0.00001
  elseif ρ <= -0.272
    return -0.999999
  elseif 0. < ρ < 0.47
    return fzero(f, 0.005, 0.995)
  elseif -0.272 < ρ < 0.
    return fzero(f, -0.995, -0.005)
  end
  0.999999
end

function frankρ2θ(ρ::Float64)
  f(θ) = 1+12*(Debye(θ, 2)- Debye(θ))/θ-ρ
  if ρ > 0.00001
    return fzero(f, ρ, 100.)
  elseif ρ < -0.00001
    return fzero(f, -100., ρ)
  end
  throw(DomainError("ρ = 0 not supported"))
end
