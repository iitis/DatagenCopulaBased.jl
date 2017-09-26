

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
    z[:,i] = cdf(p, w.*sqrt(nu./U))
  end
end

"""
  g2clsubcopula(U::Vector{Float}, rho::Float)

Returns vector of data generated using clayton (assymatric) copula accoriding to
vector of data U at given pearson correlation coeficient rho. If rho is to small or
negative its value is changed
"""
function g2clsubcopula(U::Vector{Float64}, rho::Float64)
  rho = (abs(rho) > 0.35)? abs(rho): 0.35+0.1*rand()
  tau = 2/pi*asin(rho)
  theta = 2*tau/(1-tau)
  W = rand(length(U))
  U.*(W.^(-theta/(1 + theta)) - 1 + U.^theta).^(-1/theta)
end



"""
  subcopdatagen(cli::Array = [], sti::Array = [], pixel_n::Int, band_n:::Int)

Returns Matrix{Float} pixel_n x band_n being pixel_n realisations of band_n variate
random variable with gaussian marginals, clayton copula at indeces cli, student clcopula
at sti anf gaussian copula otherwise
"""
function subcopdatagen(cli::Array = [], sti::Array = [], t::Int = 500, n::Int = 30)
  covmat, cormat = covmatgen(n)
  z = gcopulagen(cormat, t)
  if cli !=[]
    for i in 2:length(cli)
      z[:,cli[i]] = g2clsubcopula(z[:,cli[i-1]], cormat[cli[i], cli[i-1]])
    end
  end
  if sti !=[]
      g2tsubcopula!(z, cormat, sti)
  end
  convertmarg!(z, Normal, [[0, sqrt(covmat[i,i])] for i in 1:n])
end
