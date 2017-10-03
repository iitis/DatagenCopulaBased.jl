# DatagenCopulabased.jl

Copula based generator of `t` realisations of `n`-variate data in a form of `t x n` matrix `U`.
Realisations of each marginal `U[:,i]` are uniformly distributed on `[0,1]`, however the interdependence between 
marginals are not simple and depends between copula. For copula definitions and properties see e.g. 
`Copula Methods in Finance`, Umberto Cherubini, Elisa Luciano, Walter Vecchiato, Wiley 2004. 

## Installation

Within Julia, run

```julia
julia> Pkg.clone("https://github.com/ZKSI/DatagenCopulaBased.jl")
```

to install the files Julia 0.6 is required.

## Functions

### Correlation matrix generation

To generate a `n x n` correlation matrix with reference correlation `rho` run:

```julia
julia> cormatgen(n::Int, rho::Float64 = 0.5, ordered = false, altersing::Bool = true)
```

If `ordered = false` matrix correlation matrix elements varies around `rho`, 
else it drops
as a distance between marginal variables risis. If `altersing = true` some elements are positive
and some negative, else all elements are positive.

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

To generate `t` realisations of `n`-variate data from Clayton copula with 
parameter `θ > 0` run

```julia
julia> claytoncopulagen(t::Int, n::Int = 2, θ::Union{Int, Float64} = 1; pearsonrho::Bool = false, reverse::Bool = false)
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

If `pearsonrho` parameter `0 > θ >= 1` is taken as a Pearson correlation coefficent. If `reversed` returns data from reversed Clayton copula, 
`claytoncopulagen(t, n, θ; reversed = true) = 1 .- claytoncopulagen(t, n, θ)`.
For justification see: 'The use of copula functions for predictive analysis of correlations between extreme storm tides',
K. Domino, T. Błachowicz, M. Ciupak, Physica A: Statistical Mechanics and its Applications 413, 489-497, 2014. 


To generate `n` - variate data given `n-1` parameters `θ_i` run:

```julia
julia> claytoncopulagen(t::Int, θ::Array{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
```

where `n = length(θ)`, here each two neighbour marginals (`i`'th and `i+1`'th) are generated from pair Clayton copula
with parameter `θ_i >= -1 ^ θ_i != 0`. If `pearsonrho` parameters `-1 > θ_i >= 1 ^ θ_i != 0` are taken as Pearson correlation coefficients.
If `reversed` returns data from reversed Clayton copula.

```julia

julia> srand(43);

julia> x = claytoncopulagen(9, [-0.9, 0.9, 1.]; pearsonrho = true)
9×4 Array{Float64,2}:
 0.180975  0.942164   0.872673   0.872673 
 0.775377  0.230724   0.340819   0.340819 
 0.888934  0.0579034  0.190519   0.190519 
 0.924876  0.0360802  0.0294198  0.0294198
 0.408278  0.461712   0.889275   0.889275 
 0.912603  0.0433313  0.0315759  0.0315759
 0.828727  0.270476   0.274191   0.274191 
 0.400537  0.469634   0.633396   0.633396 
 0.429437  0.440285   0.478058   0.478058 

julia> cor(x)
4×4 Array{Float64,2}:
  1.0       -0.945672  -0.936555  -0.936555
 -0.945672   1.0        0.890923   0.890923
 -0.936555   0.890923   1.0        1.0
 -0.936555   0.890923   1.0        1.0 
```
### Frank copula

To generate `t` realisations of `n`-variate data from Frank copula with 
parameter `θ > 0` run

```julia
julia> frankcopulagen(t::Int, n::Int, θ::Union{Int, Float64}; pearsonrho::Bool = false)
```
If `pearsonrho` parameter `0 > θ > 1` is taken as a Pearson correlation coefficent.

```julia
julia> srand(43);

julia> frankcopulagen(10, 3, 3)
10×3 Array{Float64,2}:
 0.695637  0.894693   0.99902 
 0.805495  0.319088   0.435302
 0.907882  0.408464   0.982066
 0.93598   0.378637   0.404066
 0.311446  0.14014    0.340145
 0.672887  0.405048   0.477145
 0.877766  0.221463   0.71116 
 0.306574  0.0587265  0.834648
 0.902399  0.922309   0.89469 
 0.881764  0.697738   0.637004

```

To generate `n` - variate data given `n-1` parameters `θ_i` run:

```julia
julia> frankcopulagen(t::Int, θ::Array{Float64}; pearsonrho::Bool = false)
```

where `n = length(θ)`, here each two neighbour marginals (`i`'th and `i+1`'th) are generated from pair Frank copula
with parameter `θ_i != 0`. If `pearsonrho` parameters `-1 > θ_i > 1 ^ θ_i != 0` are taken as Pearson correlation coefficients.

### Gumbel copula

To generate `t` realisations of `n`-variate data from Clayton copula with 
parameter `θ > 0` run

```julia
julia> gumbelcopulagen(t::Int, n::Int, θ::Union{Int, Float64}; pearsonrho::Bool = false, reverse::Bool = false)
```
If `pearsonrho` parameter `0 > θ > 1` is taken as a Pearson correlation 
coefficent. If `reversed` returns data from reversed Gumbel copula.
For justification see: 'The use of copula functions for modeling the risk of 
investment in shares traded on the Warsaw Stock Exchange',
K. Domino, T. Błachowicz, Physica A: Statistical Mechanics and its Applications 413, 77-85, 2014. 

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


### Product, independent copula

To generate `t` realisations of `n` variate data from product (independent) copula run:

```julia
julia> productcopula(t::Int, n::Int)
```

### Converting marginals

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
