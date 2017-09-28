
# generated data using copulas

"""

  invers_gen(x::Vector{Float64}, theta::Float64)

Returns: Vector{Float64} of data transformed using inverse of Clayton Copula
generator with parametr theta
"""
invers_gen(x::Vector{Float64}, theta::Float64) = (1 + theta.*x).^(-1/theta)

"""

  claytoncopulagen(t::Int, n::Int)

Returns: t x n Matrix{Float}, t realisations of n-variate data generated from Clayton
 copula with parameters θ = 1

```jldoctest
julia> srand(43);

julia> claytoncopulagen(10, 2)
10×2 Array{Float64,2}:
  0.325965  0.984025
  0.364814  0.484407
  0.514236  0.990846
  0.523757  0.55038
  0.204864  0.398564
  0.890124  0.916516
  0.247198  0.746308
  0.126174  0.882004
  0.462986  0.377842
  0.950937  0.934698
 ```
"""

function claytoncopulagen(t::Int, n::Int = 2)
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
  tstudentcopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], nu::Int=10)

Generates data using t-student Copula given a correlation  matrix and degrees of freedom

```jldoctest
julia> srand(43);

julia> tstudentcopulagen(10)
10×2 Array{Float64,2}:
 0.658199  0.937148
 0.718244  0.92602
 0.809521  0.0980325
 0.263068  0.222589
 0.187187  0.971109
 0.245373  0.346428
 0.452336  0.524498
 0.57113   0.272525
 0.498443  0.48082
 0.113788  0.633349
 ```
"""

function tstudentcopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]], nu::Int=10)
  y = rand(MvNormal(cormat),t)'
  z = copy(y)
  d = Chisq(nu)
  U = rand(d, size(y, 1))
  p = TDist(nu)
  for i in 1:size(cormat, 1)
    z[:,i] = cdf(p, y[:,i].*sqrt.(nu./U)./cormat[i,i])
  end
  z
end

"""
    gausscopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]])

Returns: t x n matrix of t realisations of multivariate data generated
using gaussian copula with correlation matrix - cormat

```jldoctest

julia> srand(43);

julia> gausscopulagen(10)
10×2 Array{Float64,2}:
 0.589188  0.815308
 0.708285  0.924962
 0.747341  0.156994
 0.227634  0.183116
 0.227575  0.957376
 0.271558  0.364803
 0.445691  0.52792
 0.585362  0.23135
 0.498593  0.48266
 0.190283  0.594451
```
"""

function gausscopulagen(t::Int, cormat::Matrix{Float64} = [[1. 0.5];[0.5 1.]])
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

```jldoctest
julia> srand(43);

julia> x = rand(10,2);

julia> convertmarg!(x, Normal, [[0, 1],[0, 10]])

julia> x
10×2 Array{Float64,2}:
 -0.911655    4.17328
  0.756673  -14.4472
  1.22088   -11.4823
  1.43866   -13.1053
 -0.231978  -11.2415
  1.35696     6.43914
  0.949145  -26.0172
 -0.251957  -18.9723
 -0.177808    0.54172
  1.70477    10.4192
```
"""
VecVec = Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}


function convertmarg!(U::Matrix{T}, dist, p::VecVec = [fill([0,1], size(U, 2))...];
                                          testunif::Bool = true) where T <: AbstractFloat
  d = Uniform(0,1)
  for i = 1:size(U, 2)
    pw = pvalue(ExactOneSampleKSTest(U[:,i],d))
    testunif? pw > 0.0001 || throw(AssertionError("$i marginal not uniform")): ()
    @inbounds U[:,i] = quantile(dist(p[i]...), U[:,i])
  end
end

  # generates covariance matrix

  """
    cormatgen(n::Int, ordered = false)

Returns: symmetric correlation matrix of size `n x n`. If ordered it starts with high |cov[i;j]|
as |i-j| = 1 and |cov[i;j]| drops as |i-j| rises.

```jldoctest
julia> srand(43);

julia> cormatgen(5)
5×5 Array{Float64,2}:
  1.0       -0.464674   0.352662   0.452469   0.273749
 -0.464674   1.0       -0.143304  -0.589167  -0.354571
  0.352662  -0.143304   1.0        0.643956   0.628471
  0.452469  -0.589167   0.643956   1.0        0.356678
  0.273749  -0.354571   0.628471   0.356678   1.0
```
"""

  function cormatgen(n::Int, ordered = false)
    x = ordered? claytonsubcopulagen(3*n, rand([3., -0.8], n)) : claytoncopulagen(3*n, n)
    convertmarg!(x, Weibull, [[30-28*i/n,1] for i in 1:n])
    for i in 1:n
      x[:,i] = 10*rand([-1, 1])*x[:,i]
    end
    cor(x)
  end
