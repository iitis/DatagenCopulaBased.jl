# DatagenCopulabased.jl

Copula based generator of `t` realisations of `n`-dimensions random variable. Data are returned in a form of matrix `U`: `size(U) = (t,n)`, where 
realisations of each marginal, i.e. `U[:,i]`, are uniformly distributed on `[0,1]`. Interdependence between 
marginals are determined by a given copula. See U. Cherubini, E. Luciano, W. Vecchiato, 'Copula Methods in Finance', Wiley 2004. 

In terms of probabilistic the function `C: [0,1]ⁿ → [0,1]` is the 
`n`-dimensional copula if it is a joint cumulative distribution of
`n`-dimensions random variable with all marginals uniformly distributed on `[0,1]`.

This  module use following copula families to generate data:
* Elliptical copulas (Gaussian, t-Student),
* Archimedean copulas (Claytin, Frank, Gumbel, Ali-Mikhail-Haq)
* Marshal-Olkin copula.

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

### Product, independent copula

The use of the product copula means that each marginal variable is generated 
independently. 

```julia
julia> productcopula(t::Int, n::Int)
```

## Archimedean copulas

Archimedean one parameter bivariate copula `C(u₁,u₂) = φ⁻¹(φ(u₁)+φ(u₂))` is defined by using 
the continuous strictly 
decreasing generation function parametrised by `θ`, such that `φ(t): [0,1] → [0, ∞)` and `φ⁻¹(s)` is the pseudo-inverse. In `n`-variare case 
`C(u₁,..., uₙ) = φ⁻¹(φ(u₁)+...+φ(uₙ))` is also the copula, but constrains of 
the `θ` parameter are more strict, since in this case `φ⁻¹(s)` 
must be the inverse. For copula generators functions, their inverse, parameter 
range and `n`-dimensional sampling algorithms see: 
M. Hofert, 'Sampling Archimedean copulas', Computational Statistics & Data Analysis, 52 (2008), 5163-5174.

 * Clayton copula: `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ (0, ∞) ∪ (0, -1]` for `n = 2`,
 * Frank copula: `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ (0, ∞) ∪ (0, -∞)` for `n = 2`,
 * Gumbel copula `θ ∈ [1, ∞)`,
 * Ali-Mikhail-Haq copula `θ ∈ [0, 1)` for `n > 2` and  `θ ∈ [-1, 1]` for `n = 2`.

The `2`-dimensional Ali-Mikhail-Haq copula is discussed in 
P. Kumar, `Probability Distributions and Estimation
of Ali-Mikhail-Haq Copula`, Applied Mathematical Sciences, Vol. 4, 2010, no. 14, 657 - 666.

Following R. Nelsen 'An Introduction to Copulas', Springer Science & Business Media, 1999 - 216,
for bivariate Archimedean copulas `C(u₁,u₂)` data can be generated as follow:
 * draw `u₁ = rand()`,
 * define `w = ∂C(u₁, u₂)\∂u₁` and inverse `u₂ = f(w, u₁)`,
 * draw  `w = rand()`
 * return a pair u₁, u₂.

This method can be applied in practice for Clayton, Frank and Ali-Mikhail-Haq copula. If we use
this method recursively, we can get `n`-variate data with uniform marginals on 
`[0,1]`, where each neighbour pair
of marginals `uᵢ uⱼ` for `j = i+1` are draw form a bivariate subcopula with 
parameter `θᵢ`, the only condition for `θᵢ`
is such as for a corresponding bivariate copula. 

For each Archimedean copula the parameter `θ` can be determined by the expected Person `ρ` correlation coefficient for data
or a vector of expected Pearson correlation coefficients `[ρ₁, ..., ρₙ₋₁]` if data are generated from a series of bivariate subcopulas with 
parameters `[ϴ₁, ..., ϴₙ₋₁]`.

For Clayton, Gumbel and Ali-Mikhail-Haq family, reversed copula is also possible. The reversed copula is introduced by the transformation `uᵢ → 1-uᵢ`
for each marginal. For justification see: K. Domino, T. Błachowicz, M. Ciupak, 'The use of copula functions for predictive analysis of correlations between extreme storm tides',
Physica A: Statistical Mechanics and its Applications 413, 489-497, 2014, and K. Domino, T. Błachowicz, 'The use of copula functions for modeling the risk of 
investment in shares traded on the Warsaw Stock Exchange', Physica A: Statistical Mechanics and its Applications 413, 77-85, 2014.


### Clayton copula

To generate `t` realisations of `n`-variate data from Clayton copula with, parameter `θ > 0` run

```julia
julia> claytoncopulagen(t::Int, n::Int, θ::Union{Int, Float64}; pearsonrho::Bool = false, reverse::Bool = false)
```

```julia
julia> srand(43);

julia> claytoncopulagen(10, 2, 1)
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

 * If `pearsonrho = true`, uses Pearson correlation coefficent `0 > ρ > 1` instead of `θ`. 
 * If `reversed = true` returns data from reversed Clayton copula.

To generate `n` - variate data from a series of bivariate Clayton subcopulas with parameters `[ϴ₁, ..., ϴₙ₋₁]`, where `(θᵢ ≥ -1) ^ ∧ (θᵢ ≠ 0)` run:

```julia
julia> claytoncopulagen(t::Int, θ::Array{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
```

 `θᵢ` is a parameter of the following Clayton subcopula `C(uᵢ, u_{i+1})`. If `pearsonrho = true` use vector of
 Pearson correlation coefficients `[ρ₁, ..., ρₙ₋₁]`, where `(-1 > ρᵢ > 1) ∧ (ρᵢ ≠ 0)`.


```julia

julia> srand(43);

julia> julia> x = claytoncopulagen(9, [-0.9, 0.9]; pearsonrho = true)
9×3 Array{Float64,2}:
 0.180975  0.942164   0.872673 
 0.775377  0.230724   0.340819 
 0.888934  0.0579034  0.190519 
 0.924876  0.0360802  0.0294198
 0.408278  0.461712   0.889275 
 0.912603  0.0433313  0.0315759
 0.828727  0.270476   0.274191 
 0.400537  0.469634   0.633396 
 0.429437  0.440285   0.478058 

julia> convertmarg!(x, Normal)
                                                                                                                                                                            
julia> cor(x)
3×3 Array{Float64,2}:
  1.0       -0.945308  -0.924404
 -0.945308   1.0        0.887925
 -0.924404   0.887925   1.0  
```
### Frank copula

To generate `t` realisations of `n`-variate data from Frank copula with 
parameter `θ > 0` run

```julia
julia> frankcopulagen(t::Int, n::Int, θ::Union{Int, Float64}; pearsonrho::Bool = false)
```

```julia

julia> srand(43);

julia> frankcopulagen(10, 3, 3.)
10×3 Array{Float64,2}:
 0.330367   0.980024  0.197786
 0.386703   0.503187  0.784147
 0.585595   0.991504  0.711856
 0.609814   0.632656  0.853511
 0.14014    0.340145  0.179834
 0.908379   0.929926  0.876022
 0.291556   0.766821  0.938477
 0.0587265  0.834648  0.557912
 0.386399   0.304321  0.155315
 0.962869   0.950704  0.759655


```
If `pearsonrho = true` uses a Pearson correlation coefficent `0 > ρ > 1`, instead of `θ`.

To generate `n` - variate data from a series of bivariate Frank subcopulas with parameters `[ϴ₁, ..., ϴₙ₋₁]`, where `θᵢ ≠ 0` run:

```julia
julia> frankcopulagen(t::Int, θ::Array{Float64}; pearsonrho::Bool = false)
```

If `pearsonrho = true`, uses a Pearson correlation coefficent fulfilling `(-1 > ρᵢ > 1) ∧ (ρᵢ ≠ 0)`.

### Gumbel copula

To generate `t` realisations of `n`-variate data from Gumbel copula with 
parameter `θ ≥ 1` run

```julia
julia> gumbelcopulagen(t::Int, n::Int, θ::Union{Int, Float64}; pearsonrho::Bool = false, reverse::Bool = false)
```

```julia
julia> srand(43);

julia> gumbelcopulagen(10, 3, 3.5)
10×3 Array{Float64,2}:
 0.550199  0.574653   0.486977
 0.352515  0.0621575  0.072297
 0.31809   0.112819   0.375482
 0.652536  0.691707   0.645668
 0.988459  0.989946   0.986297
 0.731589  0.532971   0.678277
 0.62426   0.625661   0.851237
 0.335811  0.117504   0.329193
 0.504036  0.672722   0.561857
 0.326098  0.459547   0.117946
 ```

 * If `pearsonrho = true` uses Pearson correlation ceoficient parameter `0 > ρ > 1`.
 * If `reversed = true` returns data from reversed Gumbel copula.

### Ali-Mikhail-Haq copula


To generate `t` realisations of `n`-variate data from Ali-Mikhail-Haq copula with parameter `1 > θ > 0` run

```julia
julia> amhcopulagen(t::Int, n::Int, θ::Float64; pearsonrho::Bool = false, reverse::Bool = false)
```

```julia

julia> srand(43);

julia> amhcopulagen(10, 2, 0.5)
10×2 Array{Float64,2}:
 0.494523   0.993549
 0.266095   0.417142
 0.0669154  0.960595
 0.510007   0.541976
 0.0697899  0.292847
 0.754909   0.809849
 0.0352515  0.588425
 0.32647    0.973168
 0.352815   0.247616
 0.938565   0.918152
```

 * If `pearsonrho = true`, uses Pearson correlation coefficent `0 > ρ > 0.5` instead of `θ`. 
 * If `reversed = true` returns data from reversed Ali-Mikhail-Haq copula.

To generate `n` - variate data from a series of bivariate Ali-Mikhail-Haq subcopulas with parameters `[ϴ₁, ..., ϴₙ₋₁]`, where `1 ≥ θᵢ ≥ -1` run:

```julia
julia> amhcopulagen(t::Int, θ::Array{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
```

If `pearsonrho = true`, uses a Pearson correlation coefficent fulfilling `(0.5 ≥ ρᵢ > -0.2816)`.

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
