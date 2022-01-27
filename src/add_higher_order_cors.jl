VP = Vector{Pair{String,Vector{Int64}}}

# our algorithm

"""
    gcop2tstudent(x::Matrix{Real}, ind::Vector{Int}, ν::Int; naive::Bool = false)

Takes x the matrix of t realizations of data from Gaussian n-variate distribution.

Return the matrix of size x, where chosen subset of marginals (indexed by ind) has the
t-Student copula with ν degrees of freedom, all univariate marginals are
unchanged. If naive, performs the naive resampling.

```jldoctest

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42)

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
function gcop2tstudent(x, ind, ν; naive = false, rng = Random.GLOBAL_RNG)
  unique(ind) == ind || throw(AssertionError("indices must not repeat"))
  y = copy(x)
  Σ = cov(x)
  S = transpose(sqrt.(diag(Σ)))
  μ = mean(x, dims = 1)
  y = (y.-μ)./S
  if naive
    z = simulate_copula(size(x,1), Student_cop(cor(x[:,ind]), ν))
    y[:,ind] = quantile.(Normal(0,1), z)
  else
    U = rand(rng, Chisq(ν), size(x, 1))
    for i in ind
      z = y[:,i].*sqrt.(ν./U)
      z = cdf.(TDist(ν), z)
      y[:,i] = quantile.(Normal(0,1), z)
    end
  end
  y.*S.+μ
end

"""
    gcop2arch(x::Matrix{Real}, inds::Vector{Pair{String,Vector{Int64}}}; naive::Bool = false, notnested::Bool = false, rng = Random.GLOBAL_RNG)


Takes x the matrix of t realizations of data from Gaussian n-variate distribution.
Return a matrix of size x, where chosen subset of marginals (inds[i][2]) has an Archimedean
sub-copula (denoted by inds[i][1]), all univariate marginals are unchanged.

For example inds = ["clayton" => [1,2]] means that the subset of marginals indexed by 1,2
will be changed to the Clayton sub-copula. Inds may have more elements but marginals must
not overlap.

If naive, the naive resampling will be used. In notnested nested Archimedean copulas
will not be used.

```jldoctest

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42)

julia> x = rand(MvNormal(Σ), 6)'
6×3 Array{Float64,2}:
 -0.556027  -0.662861   -0.384124
 -0.299484   1.38993    -0.571326
 -0.468606  -0.0990787  -2.3464
  1.00331    1.43902     0.966819
  0.518149   1.55065     0.989712
 -0.886205   0.149748   -1.54419

julia> gcop2arch(x, ["clayton" => [1,2]]; naive::Bool = false, notnested::Bool = false)
6×3 Array{Float64,2}:
 -0.742443   0.424851  -0.384124
  0.211894   0.195774  -0.571326
 -0.989417  -0.299369  -2.3464
  0.157683   1.47768    0.966819
  0.154893   0.893253   0.989712
 -0.657297  -0.339814  -1.54419
```
"""
function gcop2arch(x, inds; naive = false, notnested = false, rng = Random.GLOBAL_RNG)
  testind(inds)
  S = transpose(sqrt.(diag(cov(x))))
  μ = mean(x, dims=1)
  x = (x.-μ)./S
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  for p in inds
    ind = p[2]
    v = naive ? rand(rng, size(xgauss, 1), length(ind)+1) : norm2unifind(xgauss, ind)
    if notnested | (length(ind) == 2) | naive
      θ = ρ2θ(meanΣ(corspearman(xgauss)[ind, ind]), p[1])
      x[:,ind] = arch_gen(p[1], v, θ; rng = rng)
    else
      part, ρslocal, ρglobal = getcors_advanced(xgauss[:,ind])
      ϕ = [ρ2θ(abs(ρ), p[1]) for ρ=ρslocal]
      θ = ρ2θ(abs(ρglobal), p[1])
      ind_adjusted = [ind[p] for p=part]
      x[:,ind] = nestedcopulag(p[1], part, ϕ, θ, v; rng = rng)
    end
  end
  quantile.(Normal(0,1), x).*S.+μ
end

"""
    gcop2frechet(x::Matrix{Real}, inds::Vector{Int}; naive::Bool = false, rng = Random.GLOBAL_RNG)

Takes x the matrix of t realizations of data from the Gaussian n-variate distribution.

Return the matrix of size x, where chosen subset of marginals (determined by inds) has the Frechet (one parameter)
sub-copula, all univariate marginals are unchanged. If naive, performs the naive resampling.

```jldoctest

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42)

julia> x = rand(MvNormal(Σ), 6)'
6×3 Array{Float64,2}:
 -0.556027  -0.662861   -0.384124
 -0.299484   1.38993    -0.571326
 -0.468606  -0.0990787  -2.3464
  1.00331    1.43902     0.966819
  0.518149   1.55065     0.989712
 -0.886205   0.149748   -1.54419

julia> gcop2frechet(x, [1,2])
6×3 Array{Float64,2}:
 -0.875777   -0.374723   -0.384124
  0.0960334   0.905703   -0.571326
 -0.599792   -0.0110945  -2.3464
  0.813717    1.8513      0.966819
  0.599255    1.56873     0.989712
 -0.7223     -0.172507   -1.54419
```
"""
function gcop2frechet(x, inds; naive = false, rng = Random.GLOBAL_RNG)
  unique(inds) == inds || throw(AssertionError("indices must not repeat"))
  S = transpose(sqrt.(diag(cov(x))))
  μ = mean(x, dims = 1)
  x = (x.-μ)./S
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  v = naive ? rand(rng, size(xgauss, 1), length(inds)) : norm2unifind(xgauss, inds, "frechet")
  α = meanΣ(corspearman(xgauss)[inds, inds])
  x[:,inds] = frechet(α, v; rng = rng)
  quantile.(Normal(0,1), x).*S.+μ
end

"""
    gcop2marshallolkin(x::Matrix{Real}, inds::Vector{Int}, λ1::Real = 1., λ2::Real = 1.5; naive::Bool = false, rng = Random.GLOBAL_RNG)

Takes x the matrix of t realizations of data from Gaussian n-variate distribution.

Return a matrix of size x, where chosen subset of marginals (inds) has bi-variate Marshall-Olkin
sub-copula parameterized by the free parameters λ1 and λ2.
λ12 is computed form the correlation between marginals. All univariate marginals are unchanged.
If naive, uses the naive resampling.

```jldoctest

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42)

julia> x = rand(MvNormal(Σ), 6)'
6×3 Array{Float64,2}:
 -0.556027  -0.662861   -0.384124
 -0.299484   1.38993    -0.571326
 -0.468606  -0.0990787  -2.3464
  1.00331    1.43902     0.966819
  0.518149   1.55065     0.989712
 -0.886205   0.149748   -1.54419

julia> gcop2marshallolkin(x, [1,2], 1., 1.5; naive = false)
6×3 Array{Float64,2}:
 -0.790756   0.784371  -0.384124
 -0.28088    0.338086  -0.571326
 -0.90688   -0.509684  -2.3464
  0.738628   1.71026    0.966819
  0.353654   1.19357    0.989712
 -0.867606  -0.589929  -1.54419
```
"""
function gcop2marshallolkin(x, inds, λ1 = 1., λ2 = 1.5; naive = false, rng = Random.GLOBAL_RNG)
  unique(inds) == inds || throw(AssertionError("indices must not repeat"))
  length(inds) == 2 || throw(AssertionError("not supported for |inds| > 2"))
  λ1 >= 0 || throw(DomainError("not supported for λ1 < 0"))
  λ2 >= 0 || throw(DomainError("not supported for λ2 < 0"))
  S = transpose(sqrt.(diag(cov(x))))
  μ = mean(x, dims=1)
  x = (x.-μ)./S
  xgauss = copy(x)
  x = cdf.(Normal(0,1), x)
  v = naive ? rand(rng, size(xgauss, 1), 3) : norm2unifind(xgauss, inds)
  ρ = corspearman(xgauss)[inds[1], inds[2]]
  x[:,inds] = mocopula(v, 2, τ2λ([moρ2τ(ρ)], [λ1, λ2]))
  quantile.(Normal(0,1), x).*S.+μ
end

"""
  testind(inds::Vector{Pair{String,Vector{Int64}}})

Tests if the sub copula name is supported and if their indices are disjoint.
"""
function testind(inds)
  indar = []
  for i in 1:length(inds)
    indar = vcat(indar, inds[i][2])
    inds[i][1] in ["gumbel", "clayton", "frank", "amh", "mo", "t-student", "frechet"] ||
    throw(AssertionError("$(inds[i][1]) copula family not supported"))
  end
  unique(indar) == indar || throw(AssertionError("differnt subcopulas must have different indices"))
end

"""
  norm2unifind(x::Matrix{Real}, i::Vector{Int}, cop::String)

Return uniformly distributed data from x[:,i] given a copula familly.
"""
function norm2unifind(x, i, cop = "")
  x = (cop == "frechet") ? x[:,i] : hcat(x[:,i], randn(size(x,1),1))
  a, s = eigen(cor(x))
  w = x*s./transpose(sqrt.(a))
  w[:, end] = sign(cov(x[:, 1], w[:, end]))*w[:, end]
  cdf.(Normal(0,1), w)
end

"""
  meanΣ(Σ::Matrix{Real})

Returns Real, a mean of the mean of lower diagal elements of a matrix

```jldoctest

julia> s= [1. 0.2 0.3; 0.2 1. 0.4; 0.3 0.4 1.];

julia> meanΣ(s)
0.3
```
"""
meanΣ(Σ) where T= mean(abs.(Σ[findall(tril(Σ-Matrix(I, size(Σ))).!=0)]))

"""
  mean_outer(Σ::Matrix{Real}, part::Vector{Vector{Int}})

returns a mean correlation excluding internal one is subsets determined by part
"""
function mean_outer(Σ, part)
  Σ_copy = copy(Σ)-Matrix(I, size(Σ))
  for ind=part
    Σ_copy[ind,ind] = zeros(length(ind),length(ind))
  end
  mean(filter(x->x != 0, vec(Σ_copy)))
end

"""
    parameters(x::Matrix{Real}, part::Vector{Vector{Int}})

Returns parametrization by correlation for data `x` and partition `part` for nested copulas.

"""
function parameters(Σ, part)
  ϕ = [meanΣ(Σ[ind,ind]) for ind=part]
  θ = mean_outer(Σ, part)
  ϕ, θ
end

"""
  are_parameters_good(ϕ::Vector{Real}, θ::Real)

tests sufficient nesting condition given parameters, returns bool
"""
function are_parameters_good(ϕ, θ)
  θ < minimum(filter(x->!isnan(x), ϕ))
end

"""
  Σ_theor(ϕ::Vector{Real}, θ::Real, part::Vector{Vector{Int}})

returns a matrix indicating a theoretical correlation according togiven parameters
and partition
"""
function Σ_theor(ϕ, θ, part)
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

"""
  getcors_advanced(x::Matrix{Real})

clusters data on a basis of a correlation
"""
function getcors_advanced(x)
  Σ = corspearman(x)
  var_no = size(Σ, 1)
  partitions = collect(Combinatorics.SetPartitions(1:var_no))[2:end-1] #TODO popraw first is trivial, last is nested
  params = (p->parameters(Σ, p)).(partitions)
  good_inds = findall(x->are_parameters_good(x...), params)
  filtered_params = params[good_inds]

  filtr_parts = partitions[good_inds]
  opt_val = [norm(Σ_theor(param..., fp) - Σ) for (param, fp)=zip(filtered_params, filtr_parts)]
  opt_index = findmin(opt_val)[2]
  ϕ, θ = filtered_params[opt_index]
  filter(x->length(x)>1,filtr_parts[opt_index]), filter(x->!isnan(x), ϕ), θ
end
