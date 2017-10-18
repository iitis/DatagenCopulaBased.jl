"""
  rand2cop(u1::Vector{Float64}, θ::Union{Int, Float64}, copula::String)

Returns vector of data generated using copula::String given vector of uniformly
distributed u1 and copula parameter θ.
"""

function rand2cop(u1::Vector{Float64}, θ::Union{Int, Float64}, copula::String)
  w = rand(length(u1))
  if copula == "clayton"
    return (u1.^(-θ).*(w.^(-θ/(1+θ))-1)+1).^(-1/θ)
  elseif copula == "frank"
    return -1/θ*log.(1+(w*(1-exp(-θ)))./(w.*(exp.(-θ*u1)-1)-exp.(-θ*u1)))
  elseif copula == "amh"
    a = 1-u1
    b = 1-θ.*(1+2*a.*w)+2*θ^2*a.^2.*w
    c = 1-θ.*(2-4*w+4*a.*w)+θ.^2.*(1-4*a.*w+4*a.^2.*w)
    return 2*w.*(a*θ-1).^2./(b+sqrt.(c))
  end
end

"""
  archcopulagen(t::Int, θ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of n variate data, where n = length(θ)+1.
To generate data uses Archimedean one parameter bivariate sub-copulas with parameters θᵢ ≠ 0 for each
neighbour marginals (i'th and i+1'th). If cor == "pearson", parameters
are Pearson correlation coefficents fulfilling

```jldoctest
julia> srand(43);

julia> archcopulagen(10, [4., 11.], "frank")
10×3 Array{Float64,2}:
 0.180975  0.386303   0.879254
 0.775377  0.247895   0.144803
 0.888934  0.426854   0.772457
 0.924876  0.395564   0.223155
 0.408278  0.139002   0.142997
 0.912603  0.901252   0.949828
 0.828727  0.0295759  0.0897796
 0.400537  0.0337673  0.27872
 0.429437  0.462771   0.425435
 0.955881  0.953623   0.969038
```
"""

function archcopulagen(t::Int, θ::Vector{Float64}, copula::String; rev::Bool = false,
                                                                   cor::String = "")
  if ((cor == "pearson") | (cor == "kendall"))
    θ = map(i -> usebivρ(i, copula, cor), θ)
  else
    map(i -> testbivθ(i, copula), θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    u = hcat(u, rand2cop(u[:, i], θ[i], copula))
  end
  rev? 1-u : u
end

"""

Clayton bivariate sub-copulas with parameters (θᵢ ≥ -1) ^ ∧ (θᵢ ≠ 0).
Ali-Mikhail-Haq bivariate sub-copulas with parameters -1 ≥ θᵢ ≥ 1
Frank bivariate sub-copulas with parameters (θᵢ ≠ 0)
"""
function testbivθ(θ::Union{Float64}, copula::String)
  !(0. in θ) || throw(AssertionError("not supported for θ = 0"))
  if copula == "clayton"
    θ >= -1 || throw(AssertionError("not supported for θ < -1"))
  elseif copula == "amh"
    -1 <= θ <= 1|| throw(AssertionError("not supported for θ < -1 or > 1"))
  end
end

"""

Clayton, Frank Pearson/Kendall correlation coefficents fulfulling (-1 > θᵢ > 1) ∧ (θᵢ ≠ 0)
Ali-Mikhail-Haq Pearson correlation coefficents fulfilling -0.2816 > θᵢ >= .5.
"""

function usebivρ(ρ::Float64, copula::String, cor::String)
  if copula == "amh"
      -0.2816 < ρ <= 0.5 || throw(AssertionError("correlation coeficiant must fulfill -0.2816 < ρ <= 0.5"))
    if cor == "kendall"
      -0.18 < ρ < 1/3 || throw(AssertionError("correlation coeficiant must fulfill -0.2816 < ρ <= 1/3"))
    end
  else
    -1 < ρ < 1 || throw(AssertionError("correlation coeficiant must fulfill -1 < ρ < 1"))
    !(0. in ρ) || throw(AssertionError("not supported for ρ = 0"))
  end
  (cor == "pearson")? ρ2θ(ρ, copula): τ2θ(ρ, copula)
end


"""
  g2tsubcopula!(z::Matrix{Float}, cormat::Matrix{Float}, subn::Array{Int})

Changes data generated using gaussian copula to data generated using student
 subcopula at indices subn.
"""

function g2tsubcopula!(z::Matrix{Float64}, cormat::Matrix{Float64}, subn::Array{Int}, nu::Int = 10)
  d = Chisq(nu)
  U = rand(d, size(z, 1))
  p = TDist(nu)
  for i in subn
    w = quantile(Normal(0, cormat[i,i]), z[:,i])
    z[:,i] = cdf(p, w.*sqrt.(nu./U))
  end
end


"""
  copulamixbv(t::Int, n::Int, cli::Array, sti::Array, std::Vector)

Returns Matrix{Float} t x n of t realisations of n variate random variable with gaussian marginals
with (0, std[i]) parameters. Data have generally gaussian copula, clayton subcopula at
cli indices and tstudent copula at sti indices. Obviously 0 .< cli .<= n and  0 .< sli .<= n
"""

VVI = Vector{Vector{Int}}

function copulamixbv(t::Int, n::Int = 30, cli::VVI = [[]], fi::VVI = [[]], amhi::VVI = [[]], ti::Array = [])
  Σ = cormatgen(n, 0.8,true,true)
  z = gausscopulagen(t, Σ)
  if cli !=[]
    for i in 1:length(cli)
      j = cli[i]
      θ = ρ2θ(Σ[j[1],j[2]], "clayton")
      z[:,j[2]] = rand2cop(z[:,j[1]], θ, "clayton")
    end
    for i in 1:length(fi)
      j = fi[i]
      θ = ρ2θ(Σ[j[1],j[2]], "frank")
      z[:,j[2]] = rand2cop(z[:,j[1]], θ, "frank")
    end
    for i in 1:length(amhi)
      j = amhi[i]
      ρ = Σ[j[1],j[2]]
      θ = 1.
      #println(ρ)
      if ρ < -0.28
        θ = -1.
      elseif ρ < 0.5
        θ = ρ2θ(ρ, "amh")
      end
      #println(θ)
      z[:,j[2]] = rand2cop(z[:,j[1]], θ, "amh")
    end
  end
  (ti == [])? (): g2tsubcopula!(z, Σ, tcinds)
  z, Σ
end
