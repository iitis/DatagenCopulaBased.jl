
"""
  covmatgen(band_n::Int)

Returns: symmetric correlation and covariance matrix
"""

function covmatgen(band_n::Int)
  x = clcopulagen(3*band_n, band_n, -28/band_n, 30.)
  for i in 1:band_n
    x[:,i] = rand([-1, 1])*x[:,i]
  end
  cov(x), cor(x)
end

"""
  normcopulagen(covmat::Matrix{Float}, pixel_n::Int, band_n::Int)

Returns: matrix of pixel_n realisations of band_n variate data generated
using gaussian copula with given correlation matrix
"""

function normcopulagen(cormat::Matrix{Float64}, pixel_n::Int, band_n::Int)
  y = rand(MvNormal(cormat),pixel_n)';
  z = copy(y)
  p = Normal(0,1)
  for i in 1:band_n
    z[:,i] = cdf(p, y[:,i])
  end
  z, y
end

"""
  tcopulagen!(z::Matrix{Float}, y::Matrix{Float}, symdef::Array{Int})

Changes data generated using gaussian copula z to data generated using student
(symmetric) copula at indices symdef. y is matrix of data generated using normal
distribution for a gaussian copula
"""

function tcopulagen!(z::Matrix{Float64}, y::Matrix{Float64}, symdef::Array{Int})
  nu = 10
  d = Chisq(nu)
  U = rand(d, size(z, 1))
  p = TDist(nu)
  for i in symdef
    z[:,i] = cdf(p, y[:,i].*sqrt(nu./U))
  end
end

"""
  convmarginals(y::Matrix{Float}, covmat::Matrix{Float})

Returns matrix of data with gaussian marginals at given covariance natrix,
 from copula generated data y on uniform segment [0,1]^n where n = size(y, 2)
"""
function convmarginals(y::Matrix{Float64}, covmat::Matrix{Float64})
  x = copy(y)
  for i in 1:size(y, 2)
    d = Normal(0, sqrt(covmat[i,i]))
    x[:,i] = quantile(d, y[:,i])
  end
  x
end

"""
  clcopappend(U::Vector{Float}, rho::Float)

Returns vector of data generated using clayton (assymatric) copula accoriding to
vector of data U at given pearson correlation coeficient rho. If rho is to small or
negative its value is changed
"""
function clcopappend(U::Vector{Float64}, rho::Float64)
  rho = (abs(rho) > 0.35)? abs(rho): 0.35+0.1*rand()
  tau = 2/pi*asin(rho)
  theta = 2*tau/(1-tau)
  W = rand(length(U))
  U.*(W.^(-theta/(1 + theta)) - 1 + U.^theta).^(-1/theta)
end

"""
  datagen(cli::Array = [], sti::Array = [], pixel_n::Int, band_n:::Int)

Returns Matrix{Float} pixel_n x band_n being pixel_n realisations of band_n variate
random variable with gaussian marginals, clayton copula at indeces cli, student clcopula
at sti anf gaussian copula otherwise
"""
function subcopdatagen(cli::Array = [], sti::Array = [], pixel_n::Int = 500,
  band_n::Int = 30)
  covmat, cormat = covmatgen(band_n)
  z, y = normcopulagen(cormat, pixel_n, band_n)
  if cli !=[]
    for i in 2:length(cli)
      z[:,cli[i]] = clcopappend(z[:,cli[i-1]], cormat[cli[i], cli[i-1]])
    end
  end
  if sti !=[]
      tcopulagen!(z, y, sti)
  end
  convmarginals(z, covmat)
end
