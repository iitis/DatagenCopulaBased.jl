
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

function clcopulag(t::Int, n::Int)
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

function convertmarg!(X::Matrix{T}, dist, par::Union{Vector{Vector{Int64}}, Vector{Vector{T}}}) where T <: AbstractFloat
  for i = 1:size(X, 2)
    @inbounds X[:,i] = quantile(dist(par[i]...), X[:,i])
  end
end


function clcopulagen(t::Int, n::Int, step::Float64 = 0.01, w1 = 1.)
  X = clcopulag(t, n)
  convertmarg!(X, Weibull, [[w1+step*i,1] for i in 1:n])
  X
end


"""
  u2normal(y::Matrix{Float}, covmat::Matrix{Float})

Returns matrix of data with gaussian marginals at given covariance natrix,
 from copula generated data y on uniform segment [0,1]^n where n = size(y, 2)
"""


function u2normal(y::Matrix{Float64}, cormat::Matrix{Float64} = eye(size(y, 2)))
  x = copy(y)
  convertmarg!(x, Normal, [[0, sqrt(cormat[i,i])] for i in 1:size(cormat, 1)])
  x
end

function u2tdist(y::Matrix{Float64}, nu::Int = 10)
  x = copy(y)
  convertmarg!(x, TDist, [[nu] for i in 1:size(y, 2)])
  x
end


function u2tdist1(y::Matrix{Float64}, nu::Int = 10)
    x = copy(y)
    for i in 1:size(y, 2)
      d = TDist(nu)
      x[:,i] = quantile(d, y[:,i])
    end
    x
  end

# generates data given copula and marginal dists

"""
  tcopulagmarg(cormat::Matrix{Float64}, t::Int, nu::Int)

Returns: t x n matrix of t realisations of multivariate data generated using
t-Student copula with nu degrees of freedom ans standard normal marginals

"""

tcopulagmarg(cormat::Matrix{Float64}, t::Int, nu::Int) = u2normal(tcopulagen(cormat, t, nu))


gcopulatmarg(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10) =
  u2tdist(gcopulagen(cormat, t), nu)


function tdistdat(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10)
  d = MvTDist(nu, zeros(cormat[1,:]), cormat)
  transpose(rand(d, t))
end


normdist(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10) =
  transpose(rand(MvNormal(cormat),t))


  # generates covariance matrix

  """
    covmatgen(band_n::Int)

  Returns: symmetric correlation and covariance matrix
  """

  function covmatgen(n::Int)
    x = 10.*clcopulagen(3*n, n, -28/n, 30.)
    for i in 1:n
      x[:,i] = rand([-1, 1])*x[:,i]
    end
    cov(x), cor(x)
  end
