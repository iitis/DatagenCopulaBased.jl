"""
  frankcopulagen(t::Int, θ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of length(θ)+1n-variate data, from
Frank pairs copula. Each two neighbour marginals (i'th and i+1'th) are generated
from bivariate Frank copula with parameter θ_i != 0. If pearsonrho parameters
are Pearson correlation coefficents such that -1 > θ_i >= 1 ^ θ_i != 0.

```jldoctest
julia> srand(43);

julia> frankcopulagen(10, [4., 11.])
10×3 Array{Float64,2}:
 0.220082   0.169968   0.132016
 0.246445   0.373513   0.297198
 0.0348839  0.129379   0.173934
 0.491317   0.547426   0.835849
 0.482926   0.403088   0.563117
 0.805497   0.0817427  0.0191319
 0.899497   0.92802    0.95324
 0.125435   0.275123   0.379057
 0.612692   0.83329    0.887488
 0.74624    0.60535    0.424692

```
"""
function frankcopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false)
  u = rand(t, 1)
  !(0. in θ) || throw(AssertionError("not supported for θ parameter = 0"))
  if pearsonrho
    maximum(abs.(θ)) < 1 || throw(AssertionError("correlation must be in range (-1, 1)"))
    θ = map(Frankθ, θ)
  end
  for i in 1:length(θ)
    z = u[:,i]
    w = rand(t)
    α = θ[i]
    v = -log.((exp.(-α.*z).*(1./w-1)+exp(-α))./(1+exp.(-α.*z).*(1./w-1)))/α
    u = hcat(u, v)
  end
  u
end

"""

  claytoncopulagen(t::Int = 1000, θ::Vector{Float64}; pearsonrho, reverse)

Returns: t x n Matrix{Float}, t realisations of length(θ)+1=n-variate data generated
from Clayton pairs copula. Each neighbour marginals (i'th and i+1'th) are generated
from bivariate Clayton copula with parameter θ_i >= -1 ^ θ_i != 0.
If pearsonrho parameters are Pearson correlation coefficents such that
-1 > θ_i >= 1 ^ θ_i != 0 . If reversed returns data from reversed Clayton pairs copula.

```jldoctest
julia> srand(43);

julia> x = claytoncopulagen(9, [-0.9, 0.9, 1.]; pearsonrho = true)
9×4 Array{Float64,2}:
 0.180975  0.942164   0.872673   0.872673
 0.775377  0.230724   0.340819   0.340819
 0.888934  0.0579034  0.190519   0.190519
 0.924876  0.0360802  0.0294198  0.0294198
 0.408278  0.461712   0.889275   0.889275
 0.912603  0.0433313  0.0315759  0.0315759
 0.828727  0.270476   0.274191   0.274191
 0.400537  0.469634   0.633396   0.633396
 0.429437  0.440285   0.478058   0.478058
```
"""

function claytoncopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
  minimum(θ) >= -1 || throw(AssertionError("not supported for parameter < -1"))
  !(0. in θ) || throw(AssertionError("not supported for θ parameter = 0"))
  if pearsonrho
    maximum(θ) <= 1 || throw(AssertionError("correlation coeficient must be in range (-1,1)"))
    θ = map(claytonθ, θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    w = rand(t)
    z = u[:, i]
    u = hcat(u, z.*(w.^(-θ[i]/(1 + θ[i])) - 1 + z.^θ[i]).^(-1/θ[i]))
  end
  reverse? 1-u : u
end

"""

  amhcopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool, reverse::Bool)

Returns: t x n Matrix{Float}, t realisations of length(θ)+1=n-variate data from
Ali-Mikhail-Haq pairs copula.
Each two neighbour marginals (i'th and i+1'th) are generated from bivariate
Ali-Mikhail-Haq copula with parameters 0 > θ_i >= 1. If pearsonrho parameters
are Pearson correlation coefficents such that 0 > θ_i >= .5.
 If reversed returns data from reversed Ali-Mikhail-Haq pairs copula.

```jldoctest
julia> srand(43);

julia> amhcopulagen(10, [1, 0.3])
10×3 Array{Float64,2}:
 0.180975  0.441152   0.996646
 0.775377  0.225086   0.177177
 0.888934  0.327726   0.977642
 0.924876  0.291837   0.108233
 0.408278  0.187564   0.402945
 0.912603  0.848985   0.830843
 0.828727  0.0571042  0.471947
 0.400537  0.0758159  0.913208
 0.429437  0.527526   0.405125
 0.955881  0.919363   0.838458
```
"""

function amhcopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
  minimum(θ) > 0 || throw(AssertionError("not supported for parameter <= 0"))
  maximum(θ) <= 1 || throw(AssertionError("not supported for parameter > 1"))
  if pearsonrho
    maximum(θ) <= 0.5 || throw(AssertionError("not supported for correlation > 0.5"))
    θ = map(AMHθ, θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    w = rand(t)
    z = u[:, i]
    p = θ[i]
    a = 1-z
    b = 1-p.*(1+2*a.*w)+2*p^2*a.^2.*w
    c = 1-p.*(2-4*w+4*a.*w)+p.^2.*(1-4*a.*w+4*a.^2.*w)
    u = hcat(u, 2*w.*(a*p-1).^2./(b+sqrt.(c)))
  end
  reverse? 1-u : u
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
  g2clsubcopula(U::Vector{Float}, ρ::Float)

Returns vector of data generated using clayton (assymatric) copula accoriding to
vector of data U at given pearson correlation coeficient ρ.
"""
function g2clsubcopula(U::Vector{Float64}, ρ::Float64)
  θ = claytonθ(ρ)
  W = rand(length(U))
  U.*(W.^(-θ/(1 + θ)) - 1 + U.^θ).^(-1/θ)
end



"""
  subcopdatagen(t::Int, n::Int, cli::Array, sti::Array, std::Vector)

Returns Matrix{Float} t x n of t realisations of n variate random variable with gaussian marginals
with (0, std[i]) parameters. Data have generally gaussian copula, clayton subcopula at
cli indices and tstudent copula at sti indices. Obviously 0 .< cli .<= n and  0 .< sli .<= n
"""
function subcopdatagen(t::Int, n::Int = 30, cli::Array = [], sti::Array = [], std::Vector{Float64} = [fill(1., n)...])
  cormat = cormatgen(n)
  z = gausscopulagen(t, cormat)
  if cli !=[]
    for i in 2:length(cli)
      z[:,cli[i]] = g2clsubcopula(z[:,cli[i-1]], cormat[cli[i], cli[i-1]])
    end
  end
  if sti !=[]
      g2tsubcopula!(z, cormat, sti)
  end
  convertmarg!(z, Normal, [[0, std[i]] for i in 1:n])
  z
end
