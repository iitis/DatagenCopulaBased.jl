"""
  g2tsubcopula!(z::Matrix{Float}, cormat::Matrix{Float}, subn::Array{Int})

Changes data generated using gaussian copula to data generated using student
 subcopula at indices subn.
"""

function g2tsubcopula!(x::Matrix{Float64}, Σ::Matrix{Float64}, ind::Array{Int}, ν::Int = 10)
  U = rand(Chisq(ν), size(x, 1))
  for i in ind
    z = quantile(Normal(0, Σ[i,i]), x[:,i])
    x[:,i] = cdf.(TDist(ν), z.*sqrt.(ν./U))
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
  gcop2tstudent(x::Matrix{Float64}, ind::Vector{Int}, ν::Int)

  Takes x ∈ Rᵗˣⁿ a matrix of t realisations of data from Gaussin n-variate distribution.
  Return a matrix of size x, where chosen subset of marginals (ind) has a
   t-Student copula with ν degrees of freedom, all univariate marginals are
   unchanged

```jldoctest

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> srand(42)

julia> x = rand(MvNormal(Σ), 6)'
6×3 Array{Float64,2}:
 -0.556027  -0.662861   -0.384124
 -0.299484   1.38993    -0.571326
 -0.468606  -0.0990787  -2.3464
  1.00331    1.43902     0.966819
  0.518149   1.55065     0.989712
 -0.886205   0.149748   -1.54419

julia> gcop2tstudent(x, [1,2], 6)
6×3 Array{Float64,2}:
 -0.514449  -0.49147    -0.384124
 -0.377933   1.66254    -0.571326
 -0.430426  -0.0165044  -2.3464
  0.928668   1.50472     0.966819
  0.223439   1.12372     0.989712
 -0.710786   0.239012   -1.54419

```
"""

function gcop2tstudent(x::Matrix{Float64}, ind::Vector{Int}, ν::Int)
  xbar = mean(x, 1)
  U = rand(Chisq(ν), size(x, 1))
  y = copy(x)
  Σ = cov(x)
  for i in ind
    z = (y[:,i].-xbar[i])./Σ[i,i].*sqrt.(ν./U)
    z = cdf.(TDist(ν), z)
    y[:,i] = quantile.(Normal(xbar[i],Σ[i,i]), z)
  end
  y
end

"""
  gcop2arch(x::Matrix{Float64}, inds::VP)

Takes x ∈ Rᵗˣⁿ a matrix of t realisations of data from Gaussin n-variate distribution.
Return a matrix of size x, where chosen subset of marginals (inds[i][2]) has ana Archimedean
sub-copula (denoted by inds[i][1]), all univariate marginals are unchanged.
e.g. inds = ["clayton" => [1,2]] a subset of marginals indexed by 1,2 to such with
Clayton sub-copula

```jldoctest

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> srand(42)

julia> x = rand(MvNormal(Σ), 6)'
6×3 Array{Float64,2}:
 -0.556027  -0.662861   -0.384124
 -0.299484   1.38993    -0.571326
 -0.468606  -0.0990787  -2.3464
  1.00331    1.43902     0.966819
  0.518149   1.55065     0.989712
 -0.886205   0.149748   -1.54419

julia> gcop2arch(x, ["clayton" => [1,2]])
6×3 Array{Float64,2}:
 -0.742443   0.424851  -0.384124
  0.211894   0.195774  -0.571326
 -0.989417  -0.299369  -2.3464
  0.157683   1.47768    0.966819
  0.154893   0.893253   0.989712
 -0.657297  -0.339814  -1.54419

```
"""

function gcop2arch(x::Matrix{Float64}, inds::VP; naive = false, notnested = false)
  testind(inds)
  S = transpose(sqrt.(diag(cov(x))))
  xbar = mean(x, 1)
  x = (x.-xbar)./S
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = naive? rand(size(xgauss, 1), length(ind)+1): norm2unifind(xgauss, ind)
    if notnested | (length(ind) == 2) | naive
      θ = ρ2θ(meanΣ(corspearman(xgauss)[ind, ind]), p[1])
      x[:,ind] = copulagen(p[1], v, θ)
    else
      part, ρslocal, ρglobal = getcors_advanced(xgauss[:,ind])
      ϕ = [ρ2θ(abs(ρ), p[1]) for ρ=ρslocal]
      θ = ρ2θ(abs(ρglobal), p[1])
      println(ϕ)
      println(θ)
      ind_adjusted = [ind[p] for p=part]
      println(part)
      x[:,ind] = nestedcopulag(p[1], part, ϕ, θ, v)
    end
  end
  quantile.(Normal(0,1), x).*S.+xbar
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


function mean_outer(Σ::Matrix{Float64}, part::Vector{Vector{Int}})
  Σ_copy = copy(Σ)
  for ind=part
    Σ_copy[ind,ind] = zeros(length(ind),length(ind))
  end
  mean(filter(x->x != 0, vec(Σ_copy)))
end

"""
    parameters(x::Matrix{Float64}, part::Vector{Vector{Int}})

Returns parametrization by correlation for data `x` and partition `part` for nested copulas.


"""
function parameters(Σ::Matrix{Float64}, part::Vector{Vector{Int}})
  ϕ = [meanΣ(Σ[ind,ind]) for ind=part]
  θ = mean_outer(Σ, part)
  ϕ, θ
end

function are_parameters_good(ϕ::Vector{Float64}, θ::Float64)
  θ < minimum(filter(x->!isnan(x), ϕ))
end

function Σ_theor(ϕ::Vector{Float64}, θ::Float64, part::Vector{Vector{Int}})
  n = sum(length.(part))
  result = fill(θ, n, n)
  for (ϕind,ind)=zip(ϕ,part)
    result[ind,ind] =fill(ϕind, length(ind), length(ind))
  end
  for i=1:n
    result[i,i] = 1
  end
  result
end

function getcors_advanced(x::Matrix{Float64})
  Σ = corspearman(x)
  var_no = size(Σ, 1)
  partitions = collect(Combinatorics.SetPartitions(1:var_no))[2:end-1] #TODO popraw first is trivial, last is nested
  params = (p->parameters(Σ, p)).(partitions)
  good_inds = find(x->are_parameters_good(x...), params)
  filtered_params = params[good_inds]

  filtr_parts = partitions[good_inds]
  opt_val = [vecnorm(Σ_theor(param..., fp) - Σ) for (param, fp)=zip(filtered_params, filtr_parts)]
  opt_index = findmin(opt_val)[2]
  ϕ, θ = filtered_params[opt_index]
  filter(x->length(x)>1,filtr_parts[opt_index]), filter(x->!isnan(x), ϕ), θ
end
