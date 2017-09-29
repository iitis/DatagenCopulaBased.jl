# DatagenCopulabased.jl

Copula based generator of `t` realisations of `n`-variate data in a form of `t x n` matrix `U`.
Realisations of each marginal `U[:,i]` are uniformly distributed on `[0,1]`, however the interdependence between 
marginals are not simple and depends between copula. For copula definitions and properties see e.g. 
Copula Methods in Finance, Umberto Cherubini, Elisa Luciano, Walter Vecchiato, Wiley 2004. 

## Installation

Within Julia, run

```julia
julia> Pkg.clone("https://github.com/ZKSI/DatagenCopulaBased.jl")
```

to install the files Julia 0.6 is required.

## Functions

### Corelation matrix generation

To generate a `n x n` correlation matrix with reference correlation `rho` run:

```julia
julia> cormatgen(n::Int, rho::Float64 = 0.5, ordered = false, altersing::Bool = true)
```

If `ordered = false` matrix correlation matrix elements varies arround `rho`, else it drops
as a distance between marginal variables risis. If `altersing = true` some elements are positive
and some negative, else all pelements are postive.

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

### Gaussian copula

Gaussian copula, with the symmetric correlation matrix `cormat`. Number of marginal variables `n` will be deduced from the
correlation matrix. If the covariance matrix with some diagonal entries other than `1.` it will be converted into a correlation matrix, if symmetric.
By default `cormat = [[1. 0.5];[0.5 1.]]`.

```julia
julia> gausscopulagen(t::Int, cormat::Matrix{Float64})
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

t-Student copula, with `nu` degrees of freedom and `cormat` symmetric correlation matrix, see Gaussian copula for details. By default `cormat = [[1. 0.5];[0.5 1.]]` and `nu = 10`


```julia
julia> tstudentcopulagen(t::Int, cormat::Matrix{Float64}, nu::Int)
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
 
### Clayton copula

To generate `t` realisations of `n`-variate data from Clayton copula with paramete `θ >= 0` run

```julia
julia> claytoncopulagen(t::Int, n::Int = 2, θ::Union{INT, Float64} = 1)
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
#### Reversed Clayton copula

Generalisation of reversed bivariate Clayton copula, see The use of copula functions for predictive analysis of correlations between extreme storm tides, 
K. Domino, T. Błachowicz, M. Ciupak, Physica A: Statistical Mechanics and its Applications 413, 489-497.

To generate `t` realisations of `n`-variate data from Reversed Clayton copula with paramete `θ >= 0` run

```julia
julia> revclaytoncopulagen(t::Int, n::Int = 2, θ::Union{INT, Float64} = 1)
```

```julia

julia> revclaytoncopulagen(10)
10×2 Array{Float64,2}:
 0.674035   0.0159754 
 0.635186   0.515593
 0.485764   0.00915412
 0.476243   0.44962 
 0.795136   0.601436
 0.109876   0.0834836 
 0.752802   0.253692
 0.873826   0.117996
 0.537014   0.622158
 0.0490631  0.0653018 
```
#### Clayton subcopula

It is possible to generate `t` realistations of `n`-variate data using bivariate Clayton copula with parameter `θ_i >= -1` for each pair `U_i` and `U_{i+1}`.
Hence for each neighbouring marginals we have a Clayton subcopula. Number of marginal variables is `n = length(θ)+1`. If `usecor` sperman correlation coeficinet
array is atken as a parameter array `θ`, here `-1 <= θ <= 1`.

```julia
julia> claytonsubcopulagen(t::Int, θ::Vector{Float64}; usecor::Bool)
```

```julia
julia> srand(43);

julia> x = claytonsubcopulagen(10, [1.])
10×2 Array{Float64,2}:
 0.180975  0.441152 
 0.775377  0.225086 
 0.888934  0.327726 
 0.924876  0.291837 
 0.408278  0.187564 
 0.912603  0.848985 
 0.828727  0.0571042
 0.400537  0.0758159
 0.429437  0.527526 
 0.955881  0.919363 
 

julia> srand(43);

julia> U = claytonsubcopulagen(5000, [0.5, -0.5]; usecor = true);

julia> cor(quantile(Normal(0,1), U))
3×3 Array{Float64,2}:
  1.0        0.496167  -0.235751
  0.496167   1.0       -0.473841
 -0.235751  -0.473841   1.0 
```

For reversed Clayton subcopula run:

```julia
julia> revclaytonsubcopulagen(t::Int, θ::Vector{Float64}; usecor::Bool)
```

### Product, independent copula

To generate `t` realisations of `n` variate data from product (independent) copula run:

```julia
julia> productcopula(t::Int, n::Int)
```

### Converting marinals

To convert marginals of `U \in [0,1]^n` using one type univariate of distributions `dist` with parameters `p[i]` for `i` th marginal run:

```julia
julia> convertmarg!(U::Matrix{T}, dist::Distribution, p::Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}; testunif::Bool = true)
```

It `testunif` each marginal is tested for uniformity.

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

To convert `i` th marginal to univariate distribution `dist` with parameters array `p` run 
```julia

julia> using Distributions

julia> quantile(dist(p...), U[:,i])

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
To convert all marginals to the same `dist` with the same parameters `p` run

```
julia> using Distributions

julia> quantile(dist(p...), U)
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
