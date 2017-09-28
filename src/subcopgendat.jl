"""

  claytonsubcopulagen(t::Int = 1000, θ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from
2-d Clayton subcopulas with parameters θ_1, ..., θ_n >= -1
"""

function claytonsubcopulagen(t::Int = 1000, θ::Vector{Float64} = [1,1,1,1])
  minimum(θ) >= -1 || throw(AssertionError("$i th θ parameter < -1"))
  X = rand(t,1)
  for i in 2:length(θ)
    W = rand(t)
    U = X[:, i-1]
    X = hcat(X, U.*(W.^(-θ[i]/(1 + θ[i])) - 1 + U.^θ[i]).^(-1/θ[i]))
  end
  X
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
  g2clsubcopula(U::Vector{Float}, rho::Float)

Returns vector of data generated using clayton (assymatric) copula accoriding to
vector of data U at given pearson correlation coeficient rho.
"""
function g2clsubcopula(U::Vector{Float64}, rho::Float64)
  tau = 2/pi*asin(rho)
  theta = 2*tau/(1-tau)
  W = rand(length(U))
  U.*(W.^(-theta/(1 + theta)) - 1 + U.^theta).^(-1/theta)
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
