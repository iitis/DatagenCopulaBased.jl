"""

  claytonsubcopulagen(t::Int = 1000, θ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from
2-d Clayton subcopulas with parameters θ_1, ..., θ_{n-1} >= -1

```jldoctest
julia> srand(43);

julia> x = claytonsubcopulagen(10, [1.])
10×2 Array{Float64,2}:
 0.180975  0.441152
 0.775377  0.225086
 0.888934  0.327726
 0.924876  0.291837
 0.408278  0.187564
 0.912603  0.848985
 0.828727  0.0571042
 0.400537  0.0758159
 0.429437  0.527526
 0.955881  0.919363
```
"""

function claytonsubcopulagen(t::Int, θ::Vector{Float64} = [1.,1.,1.]; usecor::Bool = false)
  minimum(θ) >= -1 || throw(AssertionError("$i th parameter < -1"))
  if usecor
    maximum(θ) <= 1 || throw(AssertionError("$i th parameter > 1"))
    θ = map(claytonθ, θ)
  end
  X = rand(t,1)
  for i in 1:length(θ)
    W = rand(t)
    U = X[:, i]
    X = hcat(X, U.*(W.^(-θ[i]/(1 + θ[i])) - 1 + U.^θ[i]).^(-1/θ[i]))
  end
  X
end

"""

  revclaytonsubcopulagen(t::Int = 1000, θ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from
2-d reversed Clayton subcopulas with parameters θ_1, ..., θ_{n-1} >= -1

"""
revclaytonsubcopulagen(t::Int, θ::Vector{Float64} = [1.,1.,1.]; usecor::Bool = false) =
  ones(t, length(θ)+1) - claytonsubcopulagen(t, θ; usecor=usecor)

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
