# DatagenCopulaBased.jl

[![Build Status](https://travis-ci.org/ZKSI/DatagenCopulaBased.jl.svg?branch=master)](https://travis-ci.org/ZKSI/DatagenCopulaBased.jl)
[![Coverage Status](https://coveralls.io/repos/github/iitis/DatagenCopulaBased.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/DatagenCopulaBased.jl?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1423246.svg)](https://doi.org/10.5281/zenodo.1423246)

Copula based data generator. Returns data in a form of a matrix `U`: `size(U) = (t,n)` - being `t` realisations of `n`-variate random variable. Be default each marginal, i.e. `U[:,i]`, is uniformly distributed on `[0,1]`. Interdependence between
marginals is modelled by appropriate n-variate copula function, see e.g.: U. Cherubini, E. Luciano, W. Vecchiato, 'Copula Methods in Finance', Wiley 2004.

This module support following copula families:
* Elliptical copulas (Gaussian, t-Student),
* Archimedean copulas (Clayton, Frank, Gumbel, Ali-Mikhail-Haq), including nested ones,
* Frechet familly copulas (maximal, minimal, independent),
* Marshall-Olkin copulas.

## Installation

Within Julia, run

```julia
pkg> add DatagenCopulaBased
```

to install the files Julia 1.0 or higher is required.

## Elliptical copulas

We use elliptical multivariate distribution (such as Gaussian or t-Student) to
construct a copula. Suppose `F(x₁, ..., xₙ)` is a cumulative density function
(cdf.)
of such multivariate distribution, and `Fᵢ(xᵢ)` is univariate cdf. of its `i`
th marginal. Hence `uᵢ = Fᵢ(xᵢ)` is from the uniform distribution on `[0,1]`,
and the elliptical
copula is: `C(u₁, ..., uₙ) = F(F₁⁻¹(u₁), ..., Fₙ⁻¹(uₙ))`.


### Gaussian copula

```julia
julia> gausscopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.])
```

The function returns `U`: `size(U) = (t,n)` - `t` realisations of `n`-variate random variable, each marginal, i.e. `U[:,i]`, is uniformly distributed on `[0,1]` and a cross-correlation is modelled by a Gaussian copula, parametrised by the symmetric positively defined correlation matrix `Σ` with
ones on diagonals `Σᵢᵢ=1` and all elements `-1 ≤ Σᵢⱼ ≤ 1 `. Number of
marginal variables is `n = size(Σ, 1) = size(Σ, 2)`.
If the symmetric covariance matrix without ones on a diagonals is imputed, it will be converted into a
correlation matrix automatically.


```julia

julia> using Random

julia> Random.seed!(43);

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

```julia
julia> tstudentcopulagen(t::Int, Σ::Matrix{Float64} = [1. 0.5; 0.5 1.], ν::Int=10)
```

The function returns `U`: `size(U) = (t,n)` - `t` realisations of `n`-variate random variable, each marginal, i.e. `U[:,i]`, is uniformly distributed on `[0,1]` and a cross-correlation is modelled by a t-Student copula parametrised by the symmetric matrix `Σ` (with ones on diagonals as in a Gaussian case) and by a numver `ν ∈ N`.


```julia
julia> Random.seed!(43);

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
decreasing generator parametrised by `θ`, such that `φ(t): [0,1] →
[0, ∞)` and `φ⁻¹(s)` is the pseudo-inverse.

We define similarly `n`-variate Archimedean copula `C(u₁,..., uₙ) = φ⁻¹(φ(u₁)+...+φ(uₙ))`. Here constrains for the`θ` parameter are more strict, see: M. Hofert, 'Sampling Archimedean copulas', Computational Statistics & Data Analysis, 52 (2008), 5163-5174.

 * Clayton copula - keyword = "clayton": `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ [-1, 0) ∪ (0, ∞)` for `n = 2`,
 * Frank copula - keyword = "frank": `θ ∈ (0, ∞)` for `n > 2` and `θ ∈ (-∞, 0) ∪ (0, ∞)` for `n = 2`,
 * Gumbel copula - keyword = "gumbel": `θ ∈ [1, ∞)`,
 * Ali-Mikhail-Haq copula - keyword = "amh": `θ ∈ (0, 1)` for `n > 2` and  `θ ∈ [-1, 1]` for `n = 2`.

For`2`-dimensional copula generate algorithms see P. Kumar, `Probability Distributions and Estimation
of Ali-Mikhail-Haq Copula`, Applied Mathematical Sciences, Vol. 4, 2010, no. 14, 657 - 666, and R. Nelsen 'An Introduction to Copulas', Springer Science & Business Media, 1999 - 216.


To generate `t` realisations of `n`-variate data from Archimedean copula with parameter θ run

```julia
julia> archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String; rev::Bool = false, cor::String = "")
```
The function returns `U`: `size(U) = (t,n)` - `t` realisations of `n`-variate random variable, each marginal, i.e. `U[:,i]`, is uniformly distributed on `[0,1]` and a cross-correlation is modelled by corresponding Archimedean copula.

```julia
julia> Random.seed!(43);

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

 * If `cor = Kendall`, uses Kendall's τ correlation coefficients `θ`.
 * If `cor = Spearman`, uses Spearman ρ correlation coefficient instead of `θ`.
 * If `reversed = true` returns data from reversed copula.

To generate data from reversed copula:

 * Generated data from corresponding copula `[u₁, ..., uᵢ, ..., uₙ]`,
 * Perform  transformation  `∀ᵢ uᵢ → 1-uᵢ`.

For modelling justification see: K. Domino, T. Błachowicz, M. Ciupak, 'The use of copula functions for predictive analysis of correlations between extreme storm tides',
Physica A: Statistical Mechanics and its Applications 413, 489-497, 2014, and K. Domino, T. Błachowicz, 'The use of copula functions for modeling the risk of
investment in shares traded on the Warsaw Stock Exchange', Physica A: Statistical Mechanics and its Applications 413, 77-85, 2014.


### Nested Archimedean copulas

To generate `t` realisations of `∑ᵢ nᵢ + m` variate data from nested
Archimedean copulas,  McNeil, A.J., 2008. 'Sampling nested Archimedean
copulas'. Journal of Statistical Computation and Simulation 78, 567–581, run:

```julia

julia> nestedarchcopulagen(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String, m::Int = 0)

```

Here `n` is a vector of number of variates of child copulas, `ϕ` are their
parameters, `θ` is a parameter of parents copula. Here last `m` variates are
generated using parents copula only.
Only such nesting that child and parents copulas are from the same family is
supported. Nesting condition requires `0 < θ ≤ minimum(ϕ)`.


```julia

julia> Random.seed!(43);

julia> nestedarchcopulagen(10, [2,2], [2., 3.], 1.1, "clayton", 1)
10×5 Array{Float64,2}:
 0.333487  0.584206   0.970471  0.352363  0.793386
 0.249313  0.0802689  0.298697  0.46432   0.78807 
 0.765832  0.272857   0.461754  0.125465  0.661781
 0.897061  0.346811   0.745457  0.899775  0.477065
 0.387096  0.268233   0.533175  0.42922   0.294137
 0.42065   0.247676   0.641627  0.538728  0.507315
 0.598049  0.138186   0.659411  0.876095  0.19471 
 0.125968  0.0643853  0.824152  0.601356  0.662842
 0.57524   0.625373   0.688956  0.57825   0.545043
 0.96839   0.899199   0.827176  0.544107  0.862751
```

#### If `copula == "gumbel"` further nesting is supported.


To generate `t` realisations of `length(θ)+1` variate data from hierarchically
nested Gumbel copula:
`C_θₙ(... C_θ₂(C_θ₁(u₁, u₂), u₃)...,  uₙ)` run:

```julia

julia>   nestedarchcopulagen(t::Int, θ::Vector{Float64}, copula::String = "gumbel")

```

Nesting condition `1 ≤ θ_{i+1} ≤ θᵢ`

```julia

julia> Random.seed!(43);

julia> x = nestedarchcopulagen(5, [4., 3., 2.], "gumbel")
5×4 Array{Float64,2}:
 0.832902  0.915821   0.852532  0.903184 
 0.266333  0.293338   0.307899  0.0346497
 0.152431  0.0432532  0.319465  0.42015  
 0.812182  0.685689   0.721783  0.554992 
 0.252867  0.521345   0.406719  0.511759 
```

To generate `t` realisations of `∑ᵢ ∑ⱼ nᵢⱼ` variate data from double nested gumbel copula:
`C_θ(C_ϕ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_ϕₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))`
 where `lᵢ = length(n[i])` run:

```julia

julia> nestedarchcopulagen::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, ϕ::Vector{Float64}, θ₀::Float64, copula::String = "gumbel")

```

```julia
julia> Random.seed!(43);

julia> x = nestedarchcopulagen(5, [[2,2],[2]], [[3., 2.], [4.]], [1.5, 2.1], 1.2, "gumbel")
5×6 Array{Float64,2}:
0.464403  0.711722   0.883035   0.896706   0.888614   0.826514
0.750596  0.768193   0.0659561  0.0252472  0.996014   0.989127
0.825211  0.712079   0.581356   0.507739   0.882675   0.84959
0.276326  0.0827071  0.240836   0.434629   0.0184611  0.031363
0.208422  0.504727   0.27561    0.639089   0.481855   0.573715

```

### Chain of bivariate Archimedean copulas


To generate `t` realisations of `length(θ)+1` variate data, using a chain of one parameter bivariate Archimedean copulas parametrised by `θᵢ` for - i'th and i+1'th marginal:

```julia

julia> chaincopulagen(t::Int, θ::Union{Vector{Float64}, Vector{Int}}, copula::Vector{String}; rev::Bool = false, cor::String = "")

```

In other words `∀i∈[1, length(θ)]` data are generated form Archimedean copula `C_{θᵢ}(uᵢ, u_{i+1})`. Due to features of bivariate copulas, each marginal `uᵢ` is uniformly
distributed on `[0,1]`, hence we got a multivariate copula, defined by
subsequent bivariate sub-copulas. The cross-corelation between marginals `i` and `j`: `i ≠ j+1` are introduced by a chain of
bivariate copulas.

Following families are supported: "clayton", "frank" and
"amh" -  Ali-Mikhail-Haq. Conditions for `θᵢ` parameters ranges such as in corresponding
bivariate copula case.

```julia

julia> Random.seed!(43);

julia> chaincopulagen(10, [4., 11.], ["frank", "frank"])
10×3 Array{Float64,2}:
 0.180975  0.386303   0.879254 
 0.775377  0.247895   0.144803 
 0.888934  0.426854   0.772457 
 0.924876  0.395564   0.223155 
 0.408278  0.139002   0.142997 
 0.912603  0.901252   0.949828 
 0.828727  0.0295759  0.0897796
 0.400537  0.0337673  0.27872  
 0.429437  0.462771   0.425435 
 0.955881  0.953623   0.969038 
```


## Marshall-Olkin copula

To generate `t` realisations of `n`-variate data from Marshall-Olkin copula with parameter series `λ` with non-negative elements `λₛ`, run:

```julia
julia> marshallolkincopulagen(t::Int, λ::Vector{Float64}; reverse::Bool = false)
```

Number of marginals is `n = ceil(Int, log(2, length(λ)-1))`.
Parameters are ordered as follow: `λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]`
If `reversed = true`, returns data from reversed Marshall-Olkin copula , i.e. generates data `[u₁, ..., uᵢ, ..., uₙ]` from given Marshall-Olkin copula and perform transformation `∀ᵢ uᵢ → 1-uᵢ`


```julia

julia> Random.seed!(43);

julia> marshallolkincopulagen(10, [0.2, 1.2, 1.6])
10×2 Array{Float64,2}:
 0.99636   0.994344 
 0.167268  0.0619408
 0.977418  0.965093 
 0.495167  0.0247053
 0.410336  0.250159 
 0.778989  0.678064 
 0.50927   0.350059 
 0.925875  0.887095 
 0.353646  0.219006 
 0.782477  0.686799 
```

To generate data from the Marshall-Olkin copula we use algorithm presented P. Embrechts, F. Lindskog, A McNeil 'Modelling Dependence with Copulas and Applications to Risk Management', 2001
∗∗


## Frechet family copulas

To generate `t` realisation of `n` variate one parameter Frechet copula `Cf = α C_{max} + (1-α) C_{⟂}`, where `0 ≤ α ≤ 1` run:


```julia
julia> frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64})
```

```julia

julia> Random.seed!(43);

julia> frechetcopulagen(10, 2, 0.5)
10×2 Array{Float64,2}:
 0.180975  0.661781  
 0.775377  0.775377  
 0.888934  0.125437  
 0.924876  0.924876  
 0.408278  0.408278  
 0.912603  0.740184  
 0.828727  0.00463791
 0.400537  0.0288987 
 0.429437  0.429437  
 0.955881  0.851275  
```

Two parameters Frechet copula, `C = α C_{max} + β C_{min} + (1- α - β) C_{⟂}`
is supported only for `n == 2`:

```julia
julia> frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64}, β::Union{Int, Float64})
```

Here where `0 ≤ α` , where `0 ≤ β` and `α + β ≤ 1`

``` julia

julia> Random.seed!(43);

julia> frechetcopulagen(10, 2, 0.4, 0.2)
10×2 Array{Float64,2}:
 0.180975  0.661781 
 0.775377  0.775377 
 0.888934  0.125437 
 0.924876  0.924876 
 0.408278  0.591722 
 0.912603  0.740184 
 0.828727  0.171273 
 0.400537  0.0288987
 0.429437  0.429437 
 0.955881  0.851275 
 ```

### Chain of bivariate Frechet copulas


To generate `t` realisations of `length(α)+1` multivariate data using a chain two parameter bivariate Frechet copulas with parameter `αᵢ` and `βᵢ` for each neighbour (i'th and i+1'th) marginals run:


```julia

julia> chainfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))

```
In other words `∀i∈[1, length(θ)]` data are generated from following Frechet copula `C_{αᵢ,βᵢ}(uᵢ, u_{i+1})`. Due to features of bivariate copulas, each marginal `uᵢ` is uniformly
distributed on `[0,1]`, hence we got a multivariate copula, defined by subsequent bivariate sub-copulas.
The relation between marginals `i` and `j`: `i ≠ j+1` are defined by a sequence of
bivariate copulas.


```julia
julia> srand(43)

julia> chainfrechetcopulagen(10, [0.6, 0.4], [0.3, 0.5])
10×3 Array{Float64,2}:
 0.996764  0.996764  0.996764 
 0.204033  0.795967  0.204033 
 0.979901  0.979901  0.0200985
 0.120669  0.879331  0.120669 
 0.453027  0.453027  0.453027 
 0.800909  0.199091  0.800909 
 0.54892   0.54892   0.54892  
 0.933832  0.933832  0.0661679
 0.396943  0.396943  0.396943 
 0.804096  0.851275  0.955881 
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

for different methods we have different outputs:

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

### Deterministic cases

To generate a correlation matrix with constant elements run:

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
the generalisation is

```julia
julia> cormatgen_two_constant(n::Int, α::Float64, β::Float64)
```
parameters should satisfy `0 <= α <= 1` and `α > β`.

```julia
julia> cormatgen_two_constant(4, 0.5, 0.2)
4×4 Array{Float64,2}:
 1.0  0.5  0.2  0.2
 0.5  1.0  0.2  0.2
 0.2  0.2  1.0  0.2
 0.2  0.2  0.2  1.0
```
to generate Toeplitz matrix with parameter `0 <= ρ <= 1` run:

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

To generate constant matrix with noise run:

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

Analogically generate noised two constants matrix run

```julia
julia> Random.seed!(43);

julia> cormatgen_two_constant_noised(4, 0.5, 0.2)
4×4 Array{Float64,2}:
 1.0        0.314724  0.290368  0.0469922
 0.314724   1.0       0.014256  0.212183 
 0.290368   0.014256  1.0       0.238089 
 0.0469922  0.212183  0.238089  1.0   
```
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

To change a chosen marginals subset `ind` of multivariate Gaussian distributed data `x` by means of t-Student sub-copula with
a parameter `ν` run:

```julia
julia> gcop2tstudent(x::Matrix{Float64}, ind::Vector{Int}, ν::Int; naive::Bool = false)
```
all univariate marginal distributions are Gaussian and unaffected by a transformation.

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
To change a chosen marginals subset `inds[i][2]` of multivariate Gaussian distributed data `x` by means of Archimedean sub-copula of family `inds[i][1]` run:

```julia
julia> gcop2arch(x::Matrix{Float64}, inds::Vector{Pair{String,Vector{Int64}}}; naive::Bool = false, notnested::Bool = false)
```
many disjoint subsets numbered by `i` with different Archimedean sub-copulas are possible. As before all univariate marginal distributions are Gaussian and unaffected by a transformation. Named parameter `naive` indicates a use of a naive algorithm of data substitution. Named parameter `notnested` means the use of one parameter Archimedean copula instead of a nested one.

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

```

To change a chosen marginals subset `ind` of multivariate Gaussian distributed data `x` by means of Frechet maximal sub-copula:

```julia
julia> gcop2frechet(x::Matrix{Float64}, ind::Vector{Int}; naive::Bool = false)
```
all univariate marginal distributions are Gaussian and unaffected by a transformation.

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

julia> gcop2frechet(x, [1,2])
6×3 Array{Float64,2}:
 -0.875777   -0.374723   -0.384124
  0.0960334   0.905703   -0.571326
 -0.599792   -0.0110945  -2.3464
  0.813717    1.8513      0.966819
  0.599255    1.56873     0.989712
 -0.7223     -0.172507   -1.54419
```

To change a chosen marginals subset `ind` of multivariate Gaussian distributed data `x` by means of bivariate Marshall-Olkin copula:

```julia
julia> gcop2marshallolkin(x::Matrix{Float64}, ind::Vector{Int}, λ1::Float64 = 1., λ2::Float64 = 1.5; naive::Bool = false)
```
all univariate marginal distributions are Gaussian and unaffected by a transformation.
We require `length(ind) = 2` `λ1 ≧ 0` and `λ2 ≧ 0`. The parameter `λ12` is computed from expected
correlation between both changed marginals.

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

Takes matrix `X`: `size(X) = (t, n)` ie `t` realisations of `n`-dimensional random variable, with all uniform marginal univariate distributions `∀ᵢ X[:,i] ∼ Uniform(0,1)`, and convert those marginals to common distribution `d` with parameters `p[i]`

```julia
julia> convertmarg!(U::Matrix{T}, d::UnionAll, p::Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}; testunif::Bool = true)
```

If `testunif = true` each marginal is tested for uniformity.

```julia
julia> using Distributions

julia> Random.seed!(43);

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

julia> quantile.(d(p...), U[:,i])
```

```julia
julia> Random.seed!(43);

julia> U = gausscopulagen(10);

julia> quantile.(Levy(0, 1), U[:,2])
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

julia> quantile.(d(p...), U)
```

```julia
julia> julia> quantile.(Levy(0, 1), U)
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

* while reffering to `gcop2arch()`, `gcop2frechet()`, and `gcop2marshallolkin()` - cite K. Domino, A. Glos: 'Introducing higher order correlations to marginals' subset of multivariate data by means of Archimedean copulas', [arXiv:1803.07813] (https://arxiv.org/abs/1803.07813).

* while reffering to `gcop2tstudent()` - cite K. Domino: 'Multivariate cumulants in features selection and outlier detection for financial data analysis', [arXiv:1804.00541] (https://arxiv.org/abs/1804.00541).
