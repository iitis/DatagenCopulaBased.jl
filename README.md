# DatagenCopulaBased.jl

[![Coverage Status](https://coveralls.io/repos/github/iitis/DatagenCopulaBased.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/DatagenCopulaBased.jl?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7944064.svg)](https://doi.org/10.5281/zenodo.7944064)

Copula based data generator. Returns data in the form of the `t x n` matrix `U` where`t` numerates the number of realizations, and `n` numerates the number of marginals. By the copula definition each marginal `uᵢ` is uniformly distributed on the segment `[0,1]`. Realizations of such marginal would be `U[:,i]`.

Interdependence between marginals is modeled by the `n`-variate copula, see e.g.: R. B. Nelsen, 'An introduction to copulas', Springer Science \& Business Media (2007). See also K. Domino: 'Selected Methods for non-Gaussian Data Analysis', Gliwice, IITiS PAN, 2019, [arXiv:1811.10486] (https://arxiv.org/abs/1811.10486).

This module support the following copula families:
* Elliptical copulas (Gaussian, t-Student),
* Frechet copulas (maximal, minimal, independent),
* Marshall-Olkin copulas,
* Archimedean copulas (Clayton, Frank, Gumbel, Ali-Mikhail-Haq),
* Archimedean nested copulas.

## Installation

Within Julia, run

```julia
pkg> add DatagenCopulaBased
```
To install the files Julia 1.0 or higher is required.

# Sampling data

To sample `t` realisations of data from `copula::TypeOfCopula` use
```julia
julia> simulate_copula(t::Int, copula::TypeOfCopula; rng::AbstractRNG = Random.GLOBAL_RNG)
```
where `rng` is the random number genrator that can be selected.

```julia
julia> Random.seed!(43);

julia> simulate_copula(3, GaussianCopula([1. 0.5; 0.5 1.]))
3×2 Array{Float64,2}:
 0.589188  0.815308
 0.708285  0.924962
 0.747341  0.156994
```
For `simulate_copula` all mentioned below copulas are supported.

Given `U` the preallocated matrix of `Float64` it can be filled by
`size(U,1)` sample of the copula by running:

```julia
julia> simulate_copula!(U::Matrix{Float64}, copula::TypeOfCopula; rng::AbstractRNG = Random.GLOBAL_RNG)
```
For `simulate_copula!` all mentioned Archimedean copulas (including nested and the chain) as well as the Frechet and the Marshal-Olkin copulas are supported. Number of marginals `size(U,2)` in the preallocated matrix must equal to these in the copula model, else the `Assertionerror` will be raised.

```julia
julia> u = zeros(6,3)
6×3 Array{Float64,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> Random.seed!(43);

julia> c = ClaytonCopula(3, 3.)
ClaytonCopula{Float64}(3, 3.0)

julia> simulate_copula!(u, c)

julia> u
6×3 Array{Float64,2}:
 0.740919   0.936613   0.968594
 0.369025   0.698884   0.586236
 0.0701388  0.185901   0.0890538
 0.535579   0.516761   0.538476
 0.487668   0.549494   0.804122
 0.653199   0.0923366  0.387304
```

## Elliptical copulas

Elliptical copula is derived form the multivariate elliptical distribution (such as the Gaussian or the t-Student). Suppose `F(x₁, ..., xₙ)` is the Cumulative Density Function (CDF)
of such multivariate distribution, and `Fᵢ(xᵢ)` is the univariate CDF of its `i`th marginal (we assume it is continuous). Hence `uᵢ = Fᵢ(xᵢ)` is modeled by the uniform distribution on `[0,1]`. Given the elliptical multivariate distribution, the elliptical
copula is: `C(u₁, ..., uₙ) = F(F₁⁻¹(u₁), ..., Fₙ⁻¹(uₙ))`.

### The Gaussian copula

```julia
julia> GaussianCopula(Σ::Matrix{Float64})
```
The Gaussian copula is parameterized by the correlation matrix `Σ` that needs to be symmetric, positively defined and with ones on the diagonal. The number of marginals is given by the size of `Σ`.

```julia
julia> GaussianCopula([1. 0.5; 0.5 1.])
GaussianCopula{Float64}([1.0 0.5; 0.5 1.0], 2)
```

### The t-Student copula

```julia
julia> StudentCopula(Σ::Matrix{Float64}, ν::Int)
```

The t-Student copula is parameterized by the `Σ` matrix a in the Gaussian copula case, and by the integer parameter `ν > 0 ` interpreted as the number of degrees of freedom. The number of marginals is given by the size of `Σ`.

```julia
julia> StudentCopula([1. 0.5; 0.5 1.], 1)
StudentCopula{Float64}([1.0 0.5; 0.5 1.0], 1, 2)
```

## The Marshall-Olkin copula

The Marshall-Olkin copula is derived form the Marshall-Olkin exponential distribution with positively valued parameters. The Marshall-Olkin copula models the dependency between the random variables subjected to external shocks. The shock connected with the single variable is modeled there by `λₖ`, while the shock connected with two variables by `λₖₗ`, etc...

```julia
julia> MarshallOlkinCopula(λ::Vector{Float64})
```

Parameters are ordered as follow in the argument vector `λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]`, all must be non-negative. The number of marginals of such implemented Marshal-Olkin copula is `n = ceil(Int, log(2, length(λ)-1))`.

To generate data from the Marshall-Olkin copula we use algorithm presented in P. Embrechts, F. Lindskog, A McNeil 'modeling Dependence with Copulas and Applications to Risk Management', 2001.

```julia
julia> c = MarshallOlkinCopula([1., 2., 3.])
MarshallOlkinCopula{Float64}(2, [1.0, 2.0, 3.0])

julia> Random.seed!(43);

julia> 5×2 Matrix{Float64}:
 0.854724   0.821831
 0.885202   0.858624
 0.471677   0.244436
 0.834864   0.356275
 0.0661758  0.033564
```

## The Frechet copula

The two parameters Frechet copula is `C(u₁, u₂) = α C_{max}(u₁, u₂) + β C_{min}(u₁, u₂) + (1- α - β) C_{⟂}(u₁, u₂)`. Here `C_{max}(u₁, u₂)` yields maximal `1` cross-correlation, while `C_{min}(u₁, u₂)` minimal `-1` cross correlation. The `C_{min}(u₁, u₂)` is the copula only in the bivariate case.
Obviously we require `0 ≤ α ≤ 1` , where `0 ≤ β ≤ 1` and `0 ≤ 1-α - β ≤ 1`.

```julia
julia> FrechetCopula(n::Int, α::Float64, β::Float64)
```
is supported only for `n = 2`.

```julia
julia> c = FrechetCopula(2, 0.4, 0.4)
FrechetCopula{Float64}(2, 0.4, 0.4)

julia> Random.seed!(43);

julia> simulate_copula(5, c)
5×2 Matrix{Float64}:
 0.180975   0.775377
 0.924876   0.408278
 0.599463   0.400537
 0.661781   0.661781
 0.0950087  0.0950087

```

The one parameter Frechet copula `C(u₁, ..., uₙ) = α C_{max}(u₁, ..., uₙ) + (1-α) C_{⟂}(u₁, ..., uₙ)`, where `0 ≤ α ≤ 1` is supported for any `n ≥ 2`.

```julia
julia> FrechetCopula(n::Int, α::Float64)
```
```julia
julia> c = FrechetCopula(3, 0.4)
FrechetCopula{Float64}(3, 0.4, 0.0)

julia> Random.seed!(43);

julia> simulate_copula(5, c)
5×3 Matrix{Float64}:
 0.180975    0.775377   0.888934
 0.408278    0.912603   0.828727
 0.661781    0.661781   0.661781
 0.125437    0.0950087  0.130474
 0.00463791  0.0288987  0.521601
 ```

## The Archimedean copulas

The bivariate Archimedean copula `C(u₁,u₂) = φ⁻¹(φ(u₁)+φ(u₂))` is defined by the continuous strictly decreasing generator function `φ(t)` parameterized by `θ`. Such generator must fulfill `φ(t): [0,1] →[0, ∞)`.
The `n`-variate Archimedean copula can be defined analogically: `C(u₁,..., uₙ) = φ⁻¹(φ(u₁)+...+φ(uₙ))`. Here the constrains for the `θ` parameter are more strict, see: M. Hofert, 'Sampling Archimedean copulas', Computational Statistics & Data Analysis, 52 (2008), 5163-5174.

Following Archimedean copulas are supported in the module:

 * Clayton copula - parameter domain: `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ [-1, 0) ∪ (0, ∞)` for `n = 2`,
 ```julia
 julia> ClaytonCopula(n::Int, θ::Float64)
 ```
 * Frank copula - parameter domain: `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ (-∞, 0) ∪ (0, ∞)` for `n = 2`,
 ```julia
 julia> FrankCopula(n::Int, θ::Float64)
 ```
 * Gumbel copula - parameter domain: `θ ∈ [1, ∞)`,
 ```julia
 julia> GumbelCopula(n::Int, θ::Float64)
 ```
 * Ali-Mikhail-Haq copula - parameter domain: `θ ∈ (0, 1)` for `n > 2` and  `θ ∈ [-1, 1]` for `n = 2`
 ```julia
 julia> AmhCopula(n::Int, θ::Float64)
 ```

For implemented sampling algorithms see as well P. Kumar, 'Probability Distributions and Estimation
of Ali-Mikhail-Haq Copula', Applied Mathematical Sciences, Vol. 4, 2010, no. 14, 657 - 666; and R. B. Nelsen, 'An introduction to copulas', Springer Science \& Business Media (2007).


```julia
julia> c = ClaytonCopula(3, 3.)
ClaytonCopula{Float64}(3, 3.0)

julia> Random.seed!(43);

julia> simulate_copula(5, c)
5×3 Array{Float64,2}:
 0.740919   0.936613  0.968594
 0.369025   0.698884  0.586236
 0.0701388  0.185901  0.0890538
 0.535579   0.516761  0.538476
 0.487668   0.549494  0.804122  
```

The optional third empty type `<: CorrelationType` parameter is used to compute `θ` from the expected Kendall - `KendallCorrelation` or Speraman
`SpearmanCorrelation` cross-correlation.
Here only positive correlations are supported, and there are some limitations are for the Ali-Mikhail-Haq copula due to limitations on `θ` there.


```julia
julia> c = ClaytonCopula(3, 0.5, KendallCorrelation)
ClaytonCopula{Float64}(3, 2.0)

julia> x = simulate_copula(500_000, c);

julia> corkendall(x)
3×3 Array{Float64,2}:
 1.0       0.500576  0.499986
 0.500576  1.0       0.501574
 0.499986  0.501574  1.0      
```

```julia
julia> c = ClaytonCopula(3, 0.5, SpearmanCorrelation)
ClaytonCopula{Float64}(3, 1.0760904048732394)

julia> x = simulate_copula(500_000, c);

julia> corspearman(x)
3×3 Array{Float64,2}:
 1.0       0.499662  0.499637
 0.499662  1.0       0.500228
 0.499637  0.500228  1.0   
```
The reversed Gumbel, Clayton and Ali-Mikhail-Haq copulas are supported as well:

```julia
julia> GumbelCopulaRev(n::Int, θ::Float64)
```
```julia
julia> ClaytonCopulaRev(n::Int, θ::Float64)
```
```julia
julia> AmhCopulaRev(n::Int, θ::Float64)
```

The reversed copula is introduced by the following transformation  `∀ᵢ uᵢ → 1-uᵢ`. For modeling justification see: K. Domino, T. Błachowicz, M. Ciupak, 'The use of copula functions for predictive analysis of correlations between extreme storm tides',
Physica A: Statistical Mechanics and its Applications 413, 489-497, (2014); and K. Domino, T. Błachowicz, 'The use of copula functions for modeling the risk of
investment in shares traded on the Warsaw Stock Exchange', Physica A: Statistical Mechanics and its Applications 413, 77-85, (2014).

```julia
julia> c = ClaytonCopulaRev(2, 5.)
ClaytonCopulaRev{Float64}(2, 5.0)

julia> Random.seed!(43);

julia> simulate_copula(10, c)
10×2 Array{Float64,2}:
 0.246822   0.0735546
 0.0214448  0.154414
 0.453721   0.598829
 0.87328    0.91861  
 0.896485   0.899053
 0.966261   0.981044
 0.0372783  0.0100412
 0.899013   0.758491
 0.473352   0.334147
 0.0438898  0.256301
```

### The nested Archimedean copulas

The Nested Archimedean copula is
`C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ)`.
Here `θ` is the parameter of the parent copula while `ϕᵢ` is the parameter of the child copula. If `m > 0`, some random variables will be modeled by the parent copula only. The example is:

```julia
julia> NestedClaytonCopula(childred::Vector{ClaytonCopula}, m::Int, θ::Float64)
```

```julia
julia> a = ClaytonCopula(2, 3.)
ClaytonCopula{Float64}(2, 3.0)

julia> b = ClaytonCopula(2, 4.)
ClaytonCopula{Float64}(2, 4.0)

julia> NestedClaytonCopula([a,b], 0, 1.)
NestedClaytonCopula{Float64}(ClaytonCopula{Float64}[ClaytonCopula{Float64}(2, 3.0), ClaytonCopula{Float64}(2, 4.0)], 0, 1.0, 4)
```
Only the nesting within the same family is supported. The sufficient nesting condition requires parameters of the children copulas to be larger than the parameter of the parent copula. For sampling one uses the algorithm form  McNeil, A.J., 'Sampling nested Archimedean copulas', Journal of Statistical Computation and Simulation 78, 567–581 (2008).

```julia
julia> a = ClaytonCopula(2, 0.9, KendallCorrelation)
ClaytonCopula{Float64}(2, 18.000000000000004)

julia> b = NestedClaytonCopula([a], 1, .2, KendallCorrelation)
NestedClaytonCopula{Float64}(ClaytonCopula{Float64}[ClaytonCopula{Float64}(2, 18.000000000000004)], 1, 0.5, 3)

julia> x = simulate_copula(500000, b);

julia> corkendall(x)
3×3 Array{Float64,2}:
 1.0       0.899927  0.201108
 0.899927  1.0       0.201252
 0.201108  0.201252  1.0    

```

For the Gumbel copula the double nesting is supported. Double Nested copula is: `C_θ(C_ϕ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_ϕₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))`. These are in the following form.

```julia
julia> DoubleNestedGumbelCopula(children::Vector{NestedGumbelCopula}, θ)
```

```julia
julia> a = GumbelCopula(2, 2.)
GumbelCopula{Float64}(2, 2.0)

julia> b = GumbelCopula(2, 3.)
GumbelCopula{Float64}(2, 3.0)

julia> c = GumbelCopula(2, 4.)
GumbelCopula{Float64}(2, 4.0)

julia> p = NestedGumbelCopula([a,b], 0, 1.75)
NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 2.0), GumbelCopula{Float64}(2, 3.0)], 0, 1.75, 4)

julia> p1 = NestedGumbelCopula([c], 1, 1.5)
NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 4.0)], 1, 1.5, 3)

julia> gp = DoubleNestedGumbelCopula([p, p1], 1.2)
DoubleNestedGumbelCopula{Float64}(NestedGumbelCopula{Float64}[NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 2.0), GumbelCopula{Float64}(2, 3.0)], 0, 1.75, 4), NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 4.0)], 1, 1.5, 3)], 1.2, 7)


julia> Random.seed!(43);

julia> simulate_copula(2, gp)
2×7 Array{Float64,2}:
 0.103462  0.358534  0.068492  0.0914353  0.90365   0.861869  0.0716466
 0.755824  0.946489  0.745881  0.916382   0.448706  0.354352  0.676657
```

Another example

```julia
julia> a = GumbelCopula(2, .9, KendallCorrelation)
GumbelCopula{Float64}(2, 10.000000000000002)

julia> b = GumbelCopula(2, 0.8, KendallCorrelation)
GumbelCopula{Float64}(2, 5.000000000000001)

julia> p = NestedGumbelCopula([a], 1, 0.6, KendallCorrelation)
NestedGumbelCopula(GumbelCopula[GumbelCopula(2, 10.000000000000002)], 1, 2.5)

julia> p1 = NestedGumbelCopula([b], 1, 0.5, KendallCorrelation)
NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 10.000000000000002)], 1, 2.5, 3)

julia> pp = DoubleNestedGumbelCopula([p,p1], 0.1, KendallCorrelation)
DoubleNestedGumbelCopula{Float64}(NestedGumbelCopula{Float64}[NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 10.000000000000002)], 1, 2.5, 3), NestedGumbelCopula{Float64}(GumbelCopula{Float64}[GumbelCopula{Float64}(2, 4.0)], 1, 1.5, 3)], 1.1111111111111112, 6)


julia> x = simulate_copula(750_000, pp);

julia> corkendall(x)
6×6 Array{Float64,2}:
 1.0        0.90994    0.599545   0.0981701  0.0982193  0.099066
 0.90994    1.0        0.599713   0.0981766  0.0982359  0.0990366
 0.599545   0.599713   1.0        0.099002   0.0989463  0.0991464
 0.0981701  0.0981766  0.099002   1.0        0.819611   0.499707
 0.0982193  0.0982359  0.0989463  0.819611   1.0        0.499743
 0.099066   0.0990366  0.0991464  0.499707   0.499743   1.0    
```

Hierarchical nested Gumbel copula is supported as well `C_θₙ₋₁(C_θₙ₋₂( ... C_θ₂(C_θ₁(u₁, u₂), u₃) ,..., uₙ₋₁)uₙ)`. Here bivariate Gumbel copulas are nested one in the another. The most inner is the ground ... ground child one and the most outer is the ground ... ground parent one. Numbel of marginals is `n = length(θ)+1`.

```julia
julia> HierarchicalGumbelCopula(θ::Vector{Float64})
```
Here `θ` is the parameter vector, starting form the ground ... ground child one and ending on the ground ... ground parent one. Hence elements of `θ`
must be sorted in the descending order.

```julia
julia> c = HierarchicalGumbelCopula([5., 4., 3.])
HierarchicalGumbelCopula{Float64}(4, [5.0, 4.0, 3.0])

julia> Random.seed!(43);

julia> simulate_copula(3, c)
3×4 Array{Float64,2}:
 0.100353  0.207903  0.0988337  0.0431565
 0.347417  0.217052  0.223734   0.042903
 0.73617   0.347349  0.168348   0.410963
```

```julia
julia> c = HierarchicalGumbelCopula([.9, 0.7, 0.5, 0.1], KendallCorrelation)
HierarchicalGumbelCopula{Float64}(5, [10.000000000000002, 3.333333333333333, 2.0, 1.1111111111111112])

julia> x = simulate_copula(750_000, c);

julia> corkendall(x)
5×5 Array{Float64,2}:
 1.0       0.900078  0.700316   0.499483   0.100431
 0.900078  1.0       0.700184   0.499376   0.100567
 0.700316  0.700184  1.0        0.499635   0.0999502
 0.499483  0.499376  0.499635   1.0        0.0999432
 0.100431  0.100567  0.0999502  0.0999432  1.0  
```

## The chain of bivariate copulas

The chain of the bivariate copulas is determined by the sequence of bivariate copulas `C₁, C₂, ..., Cₙ₋₁` where each model the subsequent pair of marginals `Cₖ(uₖ, uₖ₊₁)`.  Hence the cross-correlation is introduced locally and decreases as the distance between marginals grows.

### The chain of bivariate Archimedean copulas

In this case, each element of the copula chain is the Archimedean copula (Clayton, Frank and Ali-Mikhail-Haq families are supported). Hence the chain is parameterized by the parameters vector `θ` (with parameters domains as in the case of bivariate copulas) and the vector of string determining the copula family. Following families are supported: Clayton - key: "clayton", Frank - key: "frank" and Ali-Mikhail-Haq - key: "amh".
The `i`th element of the vector `θ` (and the `i`th element of the string with copulas names) determine the cross-correlation between the `i`th and the `i+1`th marginal. Number of marginals is `n = length(θ)+1`

```julia
julia> ChainArchimedeanCopulas(θ::Vector{Float64}, copulas::Vector{String})
```

if one want to use one copula type, use

```julia
julia> ChainArchimedeanCopulas(θ::Vector{Float64}, copulas::String)
```

```julia
julia> ChainArchimedeanCopulas([2., 3.], ["clayton", "frank"])
ChainArchimedeanCopulas(3, [2.0, 3.0], ["clayton", "frank"])
```

```julia
julia> ChainArchimedeanCopulas([2., 3.], "frank")
ChainArchimedeanCopulas(3, [2.0, 3.0], ["frank", "frank"])

```

```julia
julia> c = ChainArchimedeanCopulas([0.7, 0.5, 0.7], "clayton", KendallCorrelation)
ChainArchimedeanCopulas{Float64}(4, [4.666666666666666, 2.0, 4.666666666666666], ["clayton", "clayton", "clayton"])

julia> x = simulate_copula(750_000, c);

julia> corkendall(x)
4×4 Array{Float64,2}:
 1.0       0.699611  0.443936  0.399355
 0.699611  1.0       0.499728  0.443053
 0.443936  0.499728  1.0       0.699855
 0.399355  0.443053  0.699855  1.0       
```

Negative correlations are supported here as well:

```julia
julia> c = ChainArchimedeanCopulas([0.7, 0.5, -0.7], "clayton", KendallCorrelation)
ChainArchimedeanCopulas{Float64}(4, [4.666666666666666, 2.0, -0.8235294117647058], ["clayton", "clayton", "clayton"])

julia> x = simulate_copula(750_000, c);

julia> corkendall(x)
4×4 Array{Float64,2}:
  1.0        0.69992    0.445151  -0.372628
  0.69992    1.0        0.500511  -0.413707
  0.445151   0.500511   1.0       -0.699821
 -0.372628  -0.413707  -0.699821   1.0    
```

### The chain of bivariate Frechet copulas


Here, each bivariate copula is the two parameters Frechet one `Cₖ = C_{αₖ,βₖ}(uₖ, uₖ₊₁)`, where `αₖ` and `βₖ` are elements of parameter vectors `α` and `β` that must be of the equal size. Number of marginals in `n = length(α) = length(β)`.

```julia
julia> c = ChainFrechetCopulas(α, β)
```

```julia
julia> c = ChainFrechetCopulas([0.2, 0.3], [0.5, 0.1])
ChainFrechetCopulas{Float64}(3, [0.2, 0.3], [0.5, 0.1])

julia> Random.seed!(43);

julia> simulate_copula(3, c)
3×3 Array{Float64,2}:
 0.828727  0.171273  0.180975
 0.400537  0.408278  0.775377
 0.429437  0.912603  0.888934
```


## Correlation matrix generation

We supply a few methods to generate a `n x n` correlation matrix `Σ`.

### Fully random cases

to generate randomly a correlation matrix run

```julia
julia> cormatgen(n::Int)
```

or

```julia
julia> cormatgen_rand(n::Int)
```

For different methods we have different outputs

```julia
julia> Random.seed!(43);

julia> cormatgen(4)
4×4 Array{Float64,2}:
 1.0       0.396865  0.339354  0.193335
 0.396865  1.0       0.887028  0.51934
 0.339354  0.887028  1.0       0.551519
 0.193335  0.51934   0.551519  1.0     

julia> cormatgen_rand(4)
4×4 Array{Float64,2}:
 1.0       0.659183  0.916879  0.486979
 0.659183  1.0       0.676167  0.808264
 0.916879  0.676167  1.0       0.731206
 0.486979  0.808264  0.731206  1.0  
```

In general the second case gives higher values of correlations.

### Deterministic cases

To generate a correlation matrix with constant elements run

```julia
julia> cormatgen_constant(n::Int, α::Float64)
```

parameter `α` should satisfy `0 <= α <= 1`

```julia
julia> cormatgen_constant(4, 0.4)
4×4 Array{Float64,2}:
 1.0  0.4  0.4  0.4
 0.4  1.0  0.4  0.4
 0.4  0.4  1.0  0.4
 0.4  0.4  0.4  1.0
```
the generalisation is with two parameters `0 <= α <= 1` and `α > β`

```julia
julia> cormatgen_two_constant(n::Int, α::Float64, β::Float64)
```

```julia
julia> cormatgen_two_constant(4, 0.5, 0.2)
4×4 Array{Float64,2}:
 1.0  0.5  0.2  0.2
 0.5  1.0  0.2  0.2
 0.2  0.2  1.0  0.2
 0.2  0.2  0.2  1.0
```
Here the first constant refer to the "nesting" of the higher
correlation for the first half of marginals.
To generate the Toeplitz matrix with parameter `0 <= ρ <= 1` run:

```julia
julia> cormatgen_toeplitz(n::Int, ρ::Float64)

julia> cormatgen_toeplitz(4, 0.5)
4×4 Array{Float64,2}:
 1.0    0.5   0.25  0.125
 0.5    1.0   0.5   0.25
 0.25   0.5   1.0   0.5  
 0.125  0.25  0.5   1.0  
```

### Partially random and partially deterministic cases

To generate constant matrix with the noise run

```julia
julia> cormatgen_constant_noised(n::Int, α::Float64; ϵ::Float64 = (1.-α)/2.)
```
where the parameter `ϵ` must satisfy `0 <= ϵ <= 1-α`

```julia
julia> Random.seed!(43);

julia> cormatgen_constant_noised(4, 0.5)
4×4 Array{Float64,2}:
 1.0       0.314724  0.590368  0.346992
 0.314724  1.0       0.314256  0.512183
 0.590368  0.314256  1.0       0.538089
 0.346992  0.512183  0.538089  1.0   
```

Analogically to generate noised two constants matrix run

```julia
julia> Random.seed!(43);

julia> cormatgen_two_constant_noised(4, 0.8, 0.2)
4×4 Array{Float64,2}:
 1.0       0.754384  0.194805  0.171162
 0.754384  1.0       0.117009  0.27321
 0.194805  0.117009  1.0       0.139476
 0.171162  0.27321   0.139476  1.0    
```

Here the first constant refer to the "nesting" of the higher
correlation for the first half of marginals.
Finally to generate noised Toeplitz matrix run:

```julia
julia> cormatgen_toeplitz_noised(n::Int, ρ::Float64; ϵ=(1-ρ)/(1+ρ)/2)
```
where the parameter `ϵ must satisfy 0 <= ϵ <= (1-ρ)/(1+ρ)`

```julia
julia> Random.seed!(43);

julia> cormatgen_toeplitz_noised(4, 0.5)
4×4 Array{Float64,2}:
 1.0        0.376483  0.310246  0.0229948
 0.376483   1.0       0.376171  0.258122
 0.310246   0.376171  1.0       0.525393
 0.0229948  0.258122  0.525393  1.0  
```

## Changes the subset of marginals of multivariate Gaussian distributed data

To change the chosen marginals subset, determined by the vector of indices`ind`, of the multivariate Gaussian distributed data `x` by means of t-Student sub-copula with
a parameter `ν` run:

```julia
julia> gcop2tstudent(x::Matrix{Float64}, ind::Vector{Int}, ν::Int; naive::Bool = false, rng = Random.GLOBAL_RNG)
```
all univariate marginal distributions will be Gaussian, hence unaffected by the transformation. The keyword `naive` means the naive resampling if true.
Custom random number generator is supported.

```julia

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42);

julia> x = Array(rand(MvNormal(Σ), 6)')
6×3 Array{Float64,2}:
 -0.556027  -0.662861   -0.384124
 -0.299484   1.38993    -0.571326
 -0.468606  -0.0990787  -2.3464  
  1.00331    1.43902     0.966819
  0.518149   1.55065     0.989712
 -0.886205   0.149748   -1.54419

julia> gcop2tstudent(x, [1,2], 6)
6×3 Array{Float64,2}:
 -0.519458  -0.498377   -0.384124
 -0.37937    1.66806    -0.571326
 -0.432902  -0.0178933  -2.3464  
  1.01216    1.50814     0.966819
  0.226484   1.12436     0.989712
 -0.727203   0.238701   -1.54419
```
To change the chosen marginals subset  of the multivariate Gaussian distributed data `x` by means of the Archimedean copula run:

```julia
julia> gcop2arch(x::Matrix{Float64}, inds::Vector{Pair{String,Vector{Int64}}}; naive::Bool = false, notnested::Bool = false, rng = Random.GLOBAL_RNG)
```

Marginals to be changed are list in `inds[i][2]`, while the corresponding Archimedean copula is determined in `inds[i][1]`.
Many disjoint subsets of marginals with different Archimedean copulas can be transformed. All univariate marginal distributions are Gaussian hence unaffected by the transformation. The keyword `naive` indicates the use of the naive data resampling if `true`. The keyword `notnested` if `true` indicates the use of one parameter Archimedean copula instead of a nested one. Custom random number generator is supported.

```julia

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42)

julia> x = Array(rand(MvNormal(Σ), 6)')
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


julia> Random.seed!(42);

julia> x = Array(rand(MvNormal(Σ), 6)');

julia> gcop2arch(x, ["gumbel" => [1,2]])
6×3 Array{Float64,2}:
0.178913   1.60797    -0.384124
0.579476   0.880272   -0.571326
-0.986662  -0.0180474  -2.3464  
1.20299    2.55397     0.966819
0.857086   1.86212     0.989712
-0.548206  -0.439289   -1.54419

```

To change the chosen marginals subset list in `ind` of the multivariate Gaussian distributed data `x` by means of the Frechet maximal copula run:

```julia
julia> gcop2frechet(x::Matrix{Float64}, ind::Vector{Int}; naive::Bool = false)
```
all univariate marginal distributions are Gaussian as they are unaffected by the transformation. The keyword `naive` means naive resampling if true.

```julia

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42)

julia> x = Array(rand(MvNormal(Σ), 6)');

julia> gcop2frechet(x, [1,2])
6×3 Array{Float64,2}:
 -0.875777   -0.374723   -0.384124
  0.0960334   0.905703   -0.571326
 -0.599792   -0.0110945  -2.3464
  0.813717    1.8513      0.966819
  0.599255    1.56873     0.989712
 -0.7223     -0.172507   -1.54419
```

To change the chosen marginals subset list in `ind` of themultivariate Gaussian distributed data `x` by means of the bivariate Marshall-Olkin copula run:

```julia
julia> gcop2marshallolkin(x::Matrix{Float64}, ind::Vector{Int}, λ1::Float64 = 1., λ2::Float64 = 1.5; naive::Bool = false, rng = Random.GLOBAL_RNG)
```
all univariate marginal distributions are Gaussian and unaffected by the transformation. In the keyword `naive` is `true` uses the naive resampling.
The algorithm requires `length(ind) = 2` `λ₁ ≥ 0` and `λ₂ ≥ 0`. The parameter `λ₁₂` is computed from expected correlation between both changed marginals. Custom random number generator is supported.

```julia

julia> Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1];

julia> Random.seed!(42);

julia> x = Array(rand(MvNormal(Σ), 6)');

julia> gcop2marshallolkin(x, [1,2])
6×3 Array{Float64,2}:
 -0.790756   0.784371  -0.384124
 -0.28088    0.338086  -0.571326
 -0.90688   -0.509684  -2.3464  
  0.738628   1.71026    0.966819
  0.353654   1.19357    0.989712
 -0.867606  -0.589929  -1.54419
```
## Helpers

### Converting marginals

Takes matrix `X`: `size(X) = (t, n)` ie `t` realisations of `n`-dimensional random variable, with all uniform marginal univariate distributions `∀ᵢ X[:,i] ∼ Uniform(0,1)`, and convert those marginals to the common distribution `d` with parameters `p[i]`

```julia
julia> convertmarg!(x::Matrix{T}, d::UnionAll, p::Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}; testunif::Bool = true)
```

If `testunif = true` each marginal is tested for uniformity.

```julia
julia> c = GaussianCopula([1. 0.5; 0.5 1.])

julia> Random.seed!(43);

julia> x = simulate_copula(10, c);


julia> convertmarg!(x, Normal, [[0, 1],[0, 10]])

julia> x
10×2 Array{Float64,2}:
10×2 Matrix{Float64}:
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

julia> quantile.(d(p...), x[:,i])
```

```julia
julia> c = GaussianCopula([1. 0.5; 0.5 1.])

julia> Random.seed!(43);

julia> x = simulate_copula(10, c);

julia> quantile.(Levy(0, 1), x[:,2])
10-element Array{Float64,1}:
  18.327904335047272
 112.72788160148863  
   0.4992650891811052
   0.5642861403809334
 350.0676959136128   
   1.2175971128674394
   2.510078079677982
   0.6980591543550244
   2.0290242635860944
   3.527994542141473
```
To convert all marginals to the same `d` with the same parameters `p` run

```
julia> using Distributions

julia> quantile.(d(p...), x)
```

```julia
julia> quantile.(Levy(0, 1), x)
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
## BigFloat implementation, developement version.

For some copulas: Marshall-Olkin, Frechet, Gumbel, Frank, Ali-Mikhail-Haq,and Nested Gumbel the BigFloat implementation is supported. However it is a development version that requires enhancement of other Julia packages.

```julia

julia> θ = BigFloat(2.);

julia> Random.seed!(43)
MersenneTwister(43)

julia> simulate_copula(3, c)
3×3 Matrix{BigFloat}:
 0.201305  0.277557  0.387158
 0.395095  0.536081  0.152753
 0.612178  0.39509   0.426144

```

# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204;

* while reffering to `gcop2arch()`, `gcop2frechet()`, and `gcop2marshallolkin()` - please cite K. Domino, A. Glos: 'Introducing higher order correlations to marginals' subset of multivariate data by means of Archimedean copulas', [arXiv:1803.07813] (https://arxiv.org/abs/1803.07813);

* while reffering to `gcop2tstudent()` - please cite K. Domino: 'Multivariate cumulants in outlier detection for financial data analysis', Physica A: Statistical Mechanics and its Applications Volume 558, 15 November 2020, 124995 (https://doi.org/10.1016/j.physa.2020.124995);

* you may also cite K. Domino: 'Selected Methods for non-Gaussian Data Analysis', Gliwice, IITiS PAN, 2019, ISBN: 978-83-926054-3-0, [arXiv:1811.10486] (https://arxiv.org/abs/1811.10486)
