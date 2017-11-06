# DatagenCopulabased.jl

Copula based generator of `t` realisations of `n`-dimensions random variable. Data are returned in a form of matrix `U`: `size(U) = (t,n)`, where 
realisations of each marginal, i.e. `U[:,i]`, are uniformly distributed on `[0,1]`. Interdependence between 
marginals are determined by a given copula. See U. Cherubini, E. Luciano, W. Vecchiato, 'Copula Methods in Finance', Wiley 2004. 

In terms of probabilistic the function `C: [0,1]ⁿ → [0,1]` is the 
`n`-dimensional copula if it is a joint cumulative distribution of
`n`-dimensions random variable with all marginals uniformly distributed on `[0,1]`.

This  module use following copula families to generate data:
* Elliptical copulas (Gaussian, t-Student),
* Archimedean copulas (Claytin, Frank, Gumbel, Ali-Mikhail-Haq), including nested ones,
* Frechet familly copulas (maximal, minimal, independent),
* Marshal-Olkin copula,
* various copula mixtures, models with different sub-copulas for different sub-sets of marginals.

## Installation

Within Julia, run

```julia
julia> Pkg.clone("https://github.com/ZKSI/DatagenCopulaBased.jl")
```

to install the files Julia 0.6 is required.

## Elliptical copulas

We use elliptical multivariate distribution (such as Gaussian or t-Student) to 
construct a copula. Suppose `F(x₁, ..., xₙ)` is a cumulative density function 
(cdf.)
of such multivariate distribution, and `Fᵢ(xᵢ)` is univariate cdf. of its `i` 
th marginal. Hence `uᵢ = Fᵢ(xᵢ)` is from the uniform distribution on `[0,1]`, 
and the elliptical 
copula is: `C(u₁, ..., uₙ) = F(F₁⁻¹(u₁), ..., Fₙ⁻¹(uₙ))`.


### Gaussian copula

Gaussian copula is parametrised by the symmetric correlation matrix `Σ` with 
diag. elements `σᵢᵢ=1` and off-diag. elements `-1 ≤ σᵢⱼ ≤ 1 `, number of 
marginal variables `n = size(Σ, 1) = size(Σ, 2)`. 
If the symmetric covariance matrix is imputed, it will be converted into a 
correlation matrix automatically.


```julia
julia> gausscopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.])
```

```julia

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

### t-Student copula

t-Student copula is parametrised by the symmetric correlation matrix `Σ` (as in the Gaussian copula case) and `ν ∈ N` degrees of freedom.


```julia
julia> tstudentcopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.], ν::Int=10)
```

```julia
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


## Archimedean copulas

Archimedean one parameter bivariate copula `C(u₁,u₂) = φ⁻¹(φ(u₁)+φ(u₂))` is defined by using the continuous strictly 
decreasing generation function parametrised by `θ`, such that `φ(t): [0,1] → [0, ∞)` and `φ⁻¹(s)` is the pseudo-inverse. In `n`-variare case 
`C(u₁,..., uₙ) = φ⁻¹(φ(u₁)+...+φ(uₙ))` is also the copula, but constrains of 
the `θ` parameter are more strict, since in this case `φ⁻¹(s)` 
must be the inverse, see: M. Hofert, 'Sampling Archimedean copulas', Computational Statistics & Data Analysis, 52 (2008), 5163-5174.

 * Clayton copula - keyword = "clayton": `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ [-1, 0) ∪ (0, ∞)` for `n = 2`,
 * Frank copula - keyword = "frank": `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ (-∞, 0) ∪ (0, ∞)` for `n = 2`,
 * Gumbel copula - keyword = "gumbel": `θ ∈ [1, ∞)`,
 * Ali-Mikhail-Haq copula - keyword = "amh": `θ ∈ (0, 1)` for `n > 2` and  `θ ∈ [-1, 1]` for `n = 2`.

For`2`-dimensional copula gtenerate algorithms see P. Kumar, `Probability Distributions and Estimation
of Ali-Mikhail-Haq Copula`, Applied Mathematical Sciences, Vol. 4, 2010, no. 14, 657 - 666, and R. Nelsen 'An Introduction to Copulas', Springer Science & Business Media, 1999 - 216.


To generate `t` realisations of `n`-variate data from archimedean copula with parameter θ run

```julia
julia> archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String; rev::Bool = false, cor::String = "")
```

```julia
julia> srand(43);

julia> archcopulagen(10, 2, 1, "clayton")
10×2 Array{Float64,2}:
 0.770331  0.932834
 0.472847  0.0806845
 0.970749  0.653029
 0.622159  0.0518025
 0.402461  0.228549
 0.946375  0.842883
 0.809076  0.129038
 0.747983  0.433829
 0.374341  0.437269
 0.973066  0.910103

```

 * If `cor = kendall`, uses Kendall's τ correlation coefficentas `θ`.
 * If `cor = pearson`, uses Peasron ρ correlation coefficent instead of `θ`. 
 * If `reversed = true` returns data from reversed copula.

To generate data from reversed copula:

 * Generated data from corresponding copula `[u₁, ..., uᵢ, ..., uₙ]`,
 * Perform  transformation  `∀ᵢ uᵢ → 1-uᵢ`.

For modelling justification see: K. Domino, T. Błachowicz, M. Ciupak, 'The use of copula functions for predictive analysis of correlations between extreme storm tides',
Physica A: Statistical Mechanics and its Applications 413, 489-497, 2014, and K. Domino, T. Błachowicz, 'The use of copula functions for modeling the risk of 
investment in shares traded on the Warsaw Stock Exchange', Physica A: Statistical Mechanics and its Applications 413, 77-85, 2014.


### Nested archimedean copulas





### mixture of bivariate archimedean subcopulas

. If we use
this method recursively, we can get `n`-variate data with uniform marginals on 
`[0,1]`, where each neighbour pair
of marginals `uᵢ uⱼ` for `j = i+1` are draw form a bivariate subcopula with 
parameter `θᵢ`, the only condition for `θᵢ`
is such as for a corresponding bivariate copula. 

For each Archimedean copula the parameter `θ` can be determined by the expected Person `ρ` correlation coefficient for data
or a vector of expected Pearson correlation coefficients `[ρ₁, ..., ρₙ₋₁]` if data are generated from a series of bivariate subcopulas with 
parameters `[ϴ₁, ..., ϴₙ₋₁]`.


## Marshall-Olkin copula

To generate `t` realisations of `n`-variate data from Marshall-Olkin copula with parameter series `λ` of of non-negative elements `λₛ`, run:

```julia
julia> marshalolkincopulagen(t::Int, λ::Vector{Float64}; reverse::Bool = false)
```

Number of marginals is `n = ceil(Int, log(2, length(λ)-1))`.
Parameters are ordered as follow: `λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]`
If `reversed = true`, returns data from reversed Marshal-Olkin copula.


```julia

julia> srand(43);

julia> marshalolkincopulagen(10, [0.2, 1.2, 1.6])
10×2 Array{Float64,2}:
 0.875948   0.813807  
 0.902229   0.852105  
 0.386377   0.22781   
 0.666248   0.381651  
 0.10115    0.0283248 
 0.0666898  0.00202552
 0.99636    0.994344  
 0.0926391  0.95373   
 0.50927    0.5957    
 0.782477   0.682792  

```

To generate data from the  Marshall-Olkin copula we use algorithm presented in M. Hofert, 
'Sampling Archimedean copulas', Computational Statistics & Data Analysis, 52 (2008), 5163-5174

### Frechet familly copulas

The use of the product copula means that each marginal variable is generated 
independently. 

```julia
julia> productcopula(t::Int, n::Int)
```


## Helpers

### Correlation matrix generation

To generate a `n x n` correlation matrix `Σ`, with reference correlation `0 > ρ > 1` run:

```julia
julia> cormatgen(n::Int, ρ::Float64 = 0.5, ordered = false, altersing::Bool = true)
```

 * If `ordered = false` `Σ` elements varies around `ρ`, i.e. `σᵢⱼ ≈ ρ+δ`, else they drop
as indices differences rise, i.e. `σᵢⱼ ≳ σᵢₖ` as `|i-j| < |i-k|`. 
 * If `altersing = true`, some `σ` are positive and some negative, else `∀ᵢⱼ σᵢⱼ ≥ 0`.

```julia
julia> srand(43);

julia> cormatgen(4, 0.5)
4×4 Array{Float64,2}:
  1.0        0.566747  -0.34848   -0.413496
  0.566747   1.0       -0.496956  -0.575852
 -0.34848   -0.496956   1.0        0.612688
 -0.413496  -0.575852   0.612688   1.0
```

```julia
julia> srand(43);

julia> cormatgen(4, 0.5, true)
4×4 Array{Float64,2}:
  1.0        -0.39749   -0.422068  -0.0790561
 -0.39749     1.0        0.698496   0.380271 
 -0.422068    0.698496   1.0        0.518025 
 -0.0790561   0.380271   0.518025   1.0

```

### Converting marginals

Takes matrix `X` of realisations of `size(X,2) = n` dimensional random variable, with uniform marginals numbered by `i`, and convert those marginals to common distribution
`d` with parameters `p[i]`

```julia
julia> convertmarg!(U::Matrix{T}, d::UnionAll, p::Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}; testunif::Bool = true)
```

If `testunif = true` each marginal is tested for uniformity.

```julia
julia> using Distributions

julia> srand(43);

julia> U = gausscopulagen(10);

julia> convertmarg!(U, Normal, [[0, 1],[0, 10]])

julia> U
10×2 Array{Float64,2}:
  0.225457      8.97627 
  0.548381     14.3926
  0.666147    -10.0689                                                                                                                                                      
 -0.746662     -9.03553                                                                                                                                                     
 -0.746857     17.2101                                                                                                                                                      
 -0.608109     -3.45649 
 -0.136555      0.700419
  0.215631     -7.34409 
 -0.00352701   -0.434793
 -0.876853      2.39009 

```

To convert `i` th marginal to univariate distribution `d` with parameters array 
`p` run 
```julia

julia> using Distributions

julia> quantile(d(p...), U[:,i])

```

```julia
julia> quantile(Levy(0, 1), u[:,2])
10-element Array{Float64,1}:
  18.3279 
 112.728 
   0.499265
   0.564286
 350.068 
   1.2176
   2.51008 
   0.698059
   2.02902 
   3.52799 
```
To convert all marginals to the same `d` with the same parameters `p` run

```
julia> using Distributions

julia> quantile(d(p...), U)
```

```julia
julia> quantile(Levy(0, 1), u)
10×2 Array{Float64,2}:
 3.42919    18.3279
 7.14305   112.728 
 9.6359      0.499265
 0.687009    0.564286
 0.686835  350.068 
 0.827224    1.2176
 1.71944     2.51008 
 3.3597      0.698059
 2.18374     2.02902 
 0.582946    3.52799

```

# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.
