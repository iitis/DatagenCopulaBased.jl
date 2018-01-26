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
 0.746289   0.77815    0.800142   0.17872    0.415729    0.100993  0.289826
 0.487619   0.540306   0.578558   0.667372   0.854448    0.396435  0.252692
 0.959344   0.653536   0.685165   0.0694169  0.635065    0.943713  0.0463879
 0.949993   0.956981   0.961668   0.0512759  0.00436219  0.20456   0.19945
 0.718963   0.753662   0.777721   0.125815   0.145763    0.408557  0.379778
 0.241307   0.295647   0.338514   0.760856   0.772814    0.165766  0.544807
 0.0223038  0.0383994  0.0551596  0.960242   0.354697    0.381062  0.183945
 0.196893   0.0776412  0.103137   0.426957   0.40978     0.246162  0.831808
 0.856694   0.875834   0.888832   0.430832   0.0839215   0.610538  0.0235287
 0.574165   0.621527   0.655253   0.129092   0.0783403   0.41072   0.202428

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
    v = norm2unifind(xgauss, makeind(xgauss, p))
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
      m1, m, n = getcors(xgauss[:,ind])
      ϕ = [ρ2θ(abs.(m1[i]), p[1]) for i in 1:length(m1)]
      θ = ρ2θ(abs.(m), p[1])
      x[:,ind] = nestedcopulag(p[1], [length(s) for s in n], ϕ, θ, v)
    else
      θ = ρ2θ(Σ[ind[1], ind[2]], p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    end
  end
  x
end

"""
  ncop2arch(x::Matrix{Float64}, inds::VP)

Takes a matrix of data fram Gaussin multivariate distribution.
Return a matrix of size x, where chosen set of marginals has a copula changed to Archimedean one.
"""

function ncop2arch(x::Matrix{Float64}, inds::VP)
  testind(inds)
  S = transpose(sqrt.(diag(cov(x))))
  x = (x-mean(x, 1)[1])./S
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = norm2unifind(xgauss, makeind(xgauss, p))
    m1, m, n = getcors(xgauss[:,ind])
    ϕ = [ρ2θ(abs.(m1[i]), p[1]) for i in 1:length(m1)]
    θ = ρ2θ(abs.(m), p[1])
    x[:,ind] = nestedcopulag(p[1], [length(s) for s in n], ϕ, θ, v)
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
  makeind(x::Matrix{Float64}, ind::Pair{String,Vector{Int64}})

Returns multiindex of chosen marginals and those most correlated with chosen marginals.
"""

function makeind(x::Matrix{Float64}, ind::Pair{String,Vector{Int64}})
  i = ind[2]
  if ind[1] == "mo"
    l = length(ind[2])
    for p in 1+l:2^l-1
      i = vcat(i, findsimilar(transpose(x), i))
    end
  elseif ind[1] in ["gumbel", "clayton", "frank", "amh"]
    i = vcat(i, findsimilar(transpose(x), i))
  end
  i
end

"""
  findsimilar(x::Matrix{Float64}, ind::Vector{Int})

Returns Array{Int64,1}, an index of most simillar vector to those indexed by ind from x

```jldoctest

julia> x = [0.1 0.2 0.3 0.4; 0.2 0.3 0.4 0.5; 0.2 0.2 0.4 0.4; 0.1 0.3 0.5 0.6]

julia> findsimilar(x, [1,2])
1-element Array{Int64,1}:
 4
```
"""
function findsimilar(x::Matrix{Float64}, ind::Vector{Int})
  maxd =Float64[]
  for i in 1:size(x,1)
    if !(i in ind)
      y = vcat(x[ind,:], transpose(x[i,:]))
      push!(maxd, sum(sch.maxdists(sch.linkage(y, "average", "correlation"))))
    else
      push!(maxd, Inf)
    end
  end
  find(maxd .== minimum(maxd))
end


"""
  norm2unifind(x::Matrix{Float64}, Σ::Matrix{Float64}, i::Vector{Int})

Given normaly distributed data x with correlation matrix Σ returns
independent uniformly distributed data based on marginals of x indexed by a given
multiindex i.
"""

function norm2unifind(x::Matrix{Float64}, i::Vector{Int})
  Σ = cor(x)
  a, s = eig(Σ[i,i])
  w = x[:, i]*s./transpose(sqrt.(a))
  w[:, end] = sign(cov(x[:, i[1]], w[:, end]))*w[:, end]
  cdf.(Normal(0,1), w)
end

"""
  getclust(x::Matrix{Float64})

Returns Array{Int} of that indicates a clusters of marginals given a data matrix

``` jldoctest
julia> srand(43)

julia> getclust(randn(4,100))
4-element Array{Int32,1}:
 1
 1
 1
 1
```
"""

function getclust(x::Matrix{Float64})
  Z=sch.linkage(x, "average", "correlation")
  clusts = sch.fcluster(Z, 1, criterion="maxclust")
  for i in 2:size(x,1)
    b = sch.fcluster(Z, i, criterion="maxclust")
    if minimum([count(b.==j) for j in 1:i]) > 1
      clusts = b
    end
  end
  clusts
end

"""
  meanΣ(Σ::Matrix{Float64})

Returns Float64, a mean of the mean of lower diagal elements of a matrix
"""
meanΣ(Σ::Matrix{Float64}) = mean(abs.(Σ[find(tril(Σ-eye(Σ)).!=0)]))

"""
  getcors(x::Matrix{Float64})

retruns Float64, Vector{Float64} and Venctor{Vector{Int}}, a general mean correlation of
data, mean correlations in each cluster and indices of clusters.
"""
function getcors(x::Matrix{Float64})
  inds = getclust(transpose(x))
  Σ = corspearman(x)
  m = meanΣ(Σ)
  k = maximum(inds)
  m1 = zeros(k)
  ind = Array{Int}[]
  for i in 1:k
    j = find(inds .==i)
    m1[i] = meanΣ(Σ[j,j])
    push!(ind, j)
  end
  m = (m < minimum(m1))? m: minimum(m1)
  m1, m, ind
end
