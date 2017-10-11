lefttail(v1::Vector{T}, v2::Vector{T}, α::T = 0.002) where T <: AbstractFloat =
        sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)

 righttail(v1::Vector{T}, v2::Vector{T}, α::T = 0.998) where T <: AbstractFloat =
         sum((v1 .> α) .* (v2 .> α))./(length(v1)*(1-α))


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

τ2λ(τ::Float64, λ::Vector{Float64}) = [λ[1],λ[2],(λ[1]+λ[2])*τ/(1-τ)]

function τ2λ(τ::Vector{Float64}, λ::Vector{Float64})
  t = τ./(1-τ)
  M = eye(3) -(ones(3,3)-eye(3)) .*t
  fm = hcat([1. 1. 0.; 0. 1. 1.; 1. 0. 1.].*t, -[1.; 1.; 1.])
  ret = map(x -> (x > 0)? (x): (0.01), inv(M)*fm*λ)
  [λ[1:3]..., ret..., λ[end]]
end

function getmoλ(λ::Vector{Float64}, ind::Vector{Int})
  n = floor(Int, log(2, length(λ)+1))
  s = collect(combinations(collect(1:n)))
  λ[[ind == a for a in s]]
end

function setmoλ!(λ::Vector{Float64}, ind::Vector{Int}, a::Float64)
  n = floor(Int, log(2, length(λ)+1))
  s = collect(combinations(collect(1:n)))
  λ[[ind == a for a in s]] = a
end


using Combinatorics
ind(k::Int, s::Vector{Vector{Int}}) = [k in s[i] for i in 1:size(s,1)]
s = collect(combinations(collect(1:3)))
s[[length(s[i]) == 2 for i in 1:size(s,1)]]
τ2λ([0.1, 0.5, 0.8], [0.1, 0.2, .3, 1.0])


function AMHθ(ρ::Union{Float64, Int})
  if ρ >= 0.5
    return 0.999999
  elseif -0.3 < ρ <0.5
    function f1!(θ, fvec)
      fvec[1] = sin(pi/2*(1 - 2*(*(1-θ[1])*(1-θ[1])log(1-θ[1]) + θ[1])/(3*θ[1]^2)))-ρ
    end
    return nlsolve(f1!, [ρ]).zero[1]
  end
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

function logseriesquantile(v::Vector{Float64}, p::Float64)
  w = logseriescdf(p)
  [findlast(w .< b) for b in v]
end
