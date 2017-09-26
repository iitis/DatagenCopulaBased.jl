
# generated data using copulas

"""

  invers_gen(x::Vector{Float64}, theta::Float64)

Returns: Vector{Float64} of data transformed using inverse of Clayton Copula
generator with parametr theta
"""
invers_gen(x::Vector{Float64}, theta::Float64) = (1 + theta.*x).^(-1/theta)

"""

  clcopulagen(t::Int, n::Int)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Clayton
 copula with parameter θ = 1
"""

function clcopulagen(t::Int, n::Int)
  theta = 1.0
  qamma_dist = Gamma(1,1/theta)
  x = rand(t)
  u = rand(t, n)
  matrix = zeros(Float64, t, n)
  for i = 1:n
    matrix[:,i] = invers_gen(-log.(u[:,i])./quantile(qamma_dist, x), theta)
  end
  matrix
end

"""
  tcopulagen(cormat::Matrix{Float}, nu::Int)

Generates data using t-student Copula given a correlation  matrix and degrees of freedom
"""

function tcopulagen(cormat::Matrix{Float64}, t::Int, nu::Int=20)
  y = rand(MvNormal(cormat),t)'
  z = copy(y)
  d = Chisq(nu)
  U = rand(d, size(y, 1))
  p = TDist(nu)
  for i in 1:size(cormat, 1)
    z[:,i] = cdf(p, y[:,i].*sqrt.(nu./U))
  end
  z
end

"""
    gcopulagen(covmat::Matrix{Float}, t::Int)

Returns: t x n matrix of t realisations of multivariate data generated
using gaussian copula with correlation matrix - cormat
"""

function gcopulagen(cormat::Matrix{Float64}, t::Int)
  y = rand(MvNormal(cormat),t)'
  z = copy(y)
  for i in 1:size(cormat, 1)
    d = Normal(0, sqrt.(cormat[i,i]))
    z[:,i] = cdf(d, y[:,i])
  end
  z
end

# transforms univariate distributions

"""
  convertmarg!(X::Matrix, dist, p::Vector{Vector})

Takes t x n matrix of t realisations of n variate data with uniform marginals and
convert marginals on [0,1] and convert i th marginals to those with distribution dist
and parameters p[i] 
"""

function convertmarg!(X::Matrix{T}, dist, p::Union{Vector{Vector{Int64}}, Vector{Vector{T}}}) where T <: AbstractFloat
  for i = 1:size(X, 2)
    @inbounds X[:,i] = quantile(dist(p[i]...), X[:,i])
  end
end

  # generates covariance matrix

  """
    covmatgen(band_n::Int)

  Returns: symmetric correlation and covariance matrix
  """

  function covmatgen(n::Int)
    x = clcopulagen(3*n, n)
    convertmarg!(x, Weibull, [[30-28*i/n,1] for i in 1:n])
    for i in 1:n
      x[:,i] = 10*rand([-1, 1])*x[:,i]
    end
    cov(x), cor(x)
  end
