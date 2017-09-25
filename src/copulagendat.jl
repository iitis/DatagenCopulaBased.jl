"""

  invers_gen(x::Vector{Float64}, theta::Float64)

Returns: Vector{Float64} of data transformed using inverse of Clayton Copula
generator with parametr theta
"""
invers_gen(x::Vector{Float64}, theta::Float64) = (1 + theta.*x).^(-1/theta)

"""

  clcopulagen(t::Int, n::Int)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Clayton
 copula with Weibull marginals
"""

function clcopulagen(t::Int, n::Int, step::Float64 = 0.01, w1 = 1.)
  theta = 1.02
  qamma_dist = Gamma(1,1/theta)
  x = rand(t)
  u = rand(t, n)
  matrix = zeros(Float64, t, n)
  for i = 1:n
    unif_ret = invers_gen(-log.(u[:,i])./quantile(qamma_dist, x), theta)
    @inbounds matrix[:,i] = quantile(Weibull(w1+step*i,1), unif_ret)
  end
  matrix
end

"""
  covmatgen(band_n::Int)

Returns: symmetric correlation and covariance matrix
"""

function cormatgen(n::Int)
  x = clcopulagen(3*n, n, -28/n, 30.)
  for i in 1:n
    x[:,i] = rand([-1, 1])*x[:,i]
  end
  cor(x)
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
  u2normal(y::Matrix{Float})

Returns matrix of multivariate data with standard gaussian marginals
"""
function u2stnormal(y::Matrix{Float64})
  x = copy(y)
  for i in 1:size(y, 2)
    d = Normal(0, 1.)
    x[:,i] = quantile(d, y[:,i])
  end
  x
end


tcopulagmarg(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10) =
  u2stnormal(tcopulagen(cormat, t, nu))


function gcopulagen(cormat::Matrix{Float64}, t::Int)
  y = rand(MvNormal(cormat),t)'
  z = copy(y)
  for i in 1:size(cormat, 1)
    d = Normal(0, sqrt.(cormat[i,i]))
    z[:,i] = cdf(d, y[:,i])
  end
  z
end

function u2tdist(y::Matrix{Float64}, nu::Int = 10)
    x = copy(y)
    for i in 1:size(y, 2)
      d = TDist(nu)
      x[:,i] = quantile(d, y[:,i])
    end
    x
  end


gcopulatmarg(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10) =
  u2tdist(gcopulagen(cormat, t), nu)


function tdistdat(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10)
  d = MvTDist(nu, zeros(cormat[1,:]), cormat)
  transpose(rand(d, t))
end


normdist(cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], t::Int = 10000, nu::Int = 10) =
  transpose(rand(MvNormal(cormat),t))
