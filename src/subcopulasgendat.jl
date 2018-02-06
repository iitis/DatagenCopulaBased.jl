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
    z[:,i] = cdf.(p, w.*sqrt.(nu./U))
  end
end

VP = Vector{Pair{String,Vector{Int64}}}

"""
  bivariatecopulamix(t::Int, Σ::Matrix{Float64}, inds::Vector{Pair{String,Vector{Int64}}})

Returns Matrix{Float} t x n of t realisations of n variate uniform random variable
given a correlation matrix. Other than Gaussian copulas subcopulas are indicated in inds =
[copulaname::String, number_of_marginal_vatiables::Vector{Int}]
following bivariate subcopulas families are available: "clayton", "frank", "amh" -- Ali-Mikhail-Haq
"""

function bivariatecopulamix(t::Int, Σ::Matrix{Float64}, inds::Vector{Pair{String,Vector{Int64}}})
  z = gausscopulagen(t, Σ)
  for p in inds
    j = p[2]
    for i in 2:length(j)
      θ = ρ2θ(Σ[j[i-1],j[i]], p[1])
      z[:,j[i]] = rand2cop(z[:,j[i-1]], θ, p[1])
    end
  end
  z
end


# our algorithm
"""
  copulamix(t::Int, Σ::Matrix{Float}, inds::VP; λ::Vector{Float} = [6., 3., 1., 15.], ν::Int = 2,
                                                a::Vector{Float} = [0.1])

Returns x ∈ [0,1]ᵗⁿ data generated from gaussian copula with given correlation matrix Σ,
and replaced by gumbel, clayton, frank, amh (Ali-Mikhail-Haq), mo ("Marshal-Olkin"),
frechet or t-student copula at given marginal indices.

Thise copulas are indicated in inds = [copulaname::String, indices of marginals::Vector{Int}].
Indices array must be disjoint for different copulas.

Additional copula parameters are supplied as a named parameters, for t-student copula: ν::Int,
for "Marshal-Olkin" λ::Vector{Float64}, for frechet copula a::Vector{Float64} = α - β

```jldoctest

julia> d = ["mo" => [1,2,3], "clayton" => [4,5,6]];

julia> srand(43);

julia> Σ = cormatgen(7);

julia> copulamix(10, Σ, d)
10×7 Array{Float64,2}:
 0.813074   0.79831   0.725495    0.502207  0.901287   0.370366   0.563981
 0.300251   0.607666  0.66557     0.916828  0.880327   0.188144   0.82885
 0.714129   0.544253  0.608241    0.709312  0.798608   0.799607   0.572187
 0.839649   0.663797  0.00979512  0.7924    0.9259     0.862745   0.0405383
 0.54756    0.783914  0.585186    0.192458  0.387663   0.570068   0.799763
 0.389949   0.452495  0.523046    0.802396  0.79415    0.362793   0.551088
 0.228135   0.200149  0.268541    0.780435  0.596068   0.445581   0.514078
 0.685846   0.299556  0.258419    0.381806  0.576986   0.376558   0.306041
 0.642151   0.940707  0.951272    0.566305  0.720493   0.0103265  0.807028
 0.0899659  0.224153  0.294587    0.076334  0.0420721  0.387212   0.520707

```
"""


function copulamix(t::Int, Σ::Matrix{Float64}, inds::VP; λ::Vector{Float64} = [6., 3., 1., 15.],
                                                ν::Int = 2, a::Vector{Float64} = [0.1])
  testind(inds)
  x = transpose(rand(MvNormal(Σ),t))
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = norm2unifind(xgauss, ind, p[1])
    if p[1] == "mo"
      length(ind) < 4 || throw(DomainError("not supported for Marshal-Olkin subcopula of number of marginals > 3"))
      map = collect(combinations(1:length(ind),2))
      τ = [corkendall(xgauss[:,ind[k[1]]], xgauss[:,ind[k[2]]]) for k in map]
      x[:,ind] = mocopula(v, length(ind), τ2λ(τ, λ))
    elseif p[1] == "frechet"
      l = length(ind)-1
      α, β = frechetρ2αβ([Σ[ind[k], ind[k+1]] for k in 1:l], a)
      x[:,ind] =fncopulagen(α, β, v)
    elseif p[1] == "t-student"
      g2tsubcopula!(x, Σ, ind, ν)
    elseif length(ind) > 2
      m1, m, inds = getcors(xgauss[:,ind], div(length(ind),2)+1)
      ϕ = [ρ2θ(abs.(m1[i]), p[1]) for i in 1:length(m1)]
      θ = ρ2θ(abs.(m), p[1])
      x[:,ind] = nestedcopulag(p[1], inds, ϕ, θ, v)
    else
      θ = ρ2θ(corspearman(xgauss[:,ind[1]], xgauss[:,ind[2]]), p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    end
  end
  x
end

"""
  gcop2arch(x::Matrix{Float64}, inds::VP)

Takes a matrix of data fram Gaussin multivariate distribution.
Return a matrix of size x, where chosen set of marginals has a copula changed to Archimedean one.
"""

function gcop2arch(x::Matrix{Float64}, inds::VP; naive = false, notnested = false)
  testind(inds)
  S = transpose(sqrt.(diag(cov(x))))
  x = (x-mean(x, 1)[1])./S
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = naive? rand(size(xgauss, 1), length(ind)+1): norm2unifind(xgauss, ind)
    if notnested | (length(ind) == 2)
      θ = ρ2θ(meanΣ(corspearman(xgauss)[ind, ind]), p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    else
      k = maximum([2, div(length(ind),2)])
      m1, m, inds = getcors(xgauss[:,ind], k)
      ϕ = [ρ2θ(abs.(m1[i]), p[1]) for i in 1:length(m1)]
      θ = ρ2θ(abs.(m), p[1])
      x[:,ind] = nestedcopulag(p[1], inds, ϕ, θ, v)
    end
  end
  quantile.(Normal(0,1), x).*S
end

"""
  testind(inds::Vector{Pair{String,Vector{Int64}}})

Tests if the sub copula name is supported and if their indices are disjoint.
"""

function testind(inds::Vector{Pair{String,Vector{Int64}}})
  indar = []
  for i in 1:length(inds)
    indar = vcat(indar, inds[i][2])
    inds[i][1] in ["gumbel", "clayton", "frank", "amh", "mo", "t-student", "frechet"] ||
    throw(AssertionError("$(inds[i][1]) copula family not supported"))
  end
  unique(indar) == indar || throw(AssertionError("differnt subcopulas must heve different indices"))
end


"""
  norm2unifind(x::Matrix{Float64}, i::Vector{Int}, cop::String)

Return uniformly distributed data from x[:,i] given a copula familly.
"""

function norm2unifind(x::Matrix{Float64}, i::Vector{Int}, cop::String = "")
  #j = setdiff(collect(1:size(x, 2)), i)
  l = (cop == "mo")? 2^(length(i))-length(i)-1: 1
  x = (cop == "frechet")? x[:,i]: hcat(x[:,i], randn(size(x,1),l))
  Σ = cor(x)
  a, s = eig(Σ)
  w = x*s./transpose(sqrt.(a))
  w[:, end] = sign(cov(x[:, 1], w[:, end]))*w[:, end]
  cdf.(Normal(0,1), w)
end


"""
  meanΣ(Σ::Matrix{Float64})

Returns Float64, a mean of the mean of lower diagal elements of a matrix

```jldoctest

julia> s= [1. 0.2 0.3; 0.2 1. 0.4; 0.3 0.4 1.];

julia> meanΣ(s)
0.3
```
"""

meanΣ(Σ::Matrix{Float64}) = mean(abs.(Σ[find(tril(Σ-eye(Σ)).!=0)]))


"""
  getcors(x::Matrix{Float64})

retruns Float64, Vector{Float64} and Venctor{Vector{Int}}, a general mean correlation of
data, mean correlations in each cluster and indices of clusters.
"""


function getcors(x::Matrix{Float64}, k::Int)
  Σ = corspearman(x)
  ind, m1 = getclust(Σ, k)
  n = size(Σ,1)
  m = (sum(Σ-eye(Σ))-sum(Σ[ind,ind]-eye(Σ[ind,ind])))/(n^2-n+length(ind)-length(ind)^2)
  [m1], m, [ind]
end


"""
  getclust(Σ::Matrix{Float64}, p::Int = 2)

Returns Array{Int} of size p that indicates a clustert of higher correlated marginals
and its correlation

``` jldoctest
julia> srand(43)

julia> getclust(cor(rand(100, 4)))

julia> ([2, 3], 0.10631347336426306)
```
"""

function getclust(Σ::Matrix{Float64}, p::Int = 2)
  x = findmax( meanΣ(Σ[inds,inds]) for inds =subsets(1:size(Σ, 1), p))
  collect(subsets(1:size(Σ, 1), p))[x[2]], x[1]
end
