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

To generate a `n x n` correlation matrix run:

```julia
julia> cormatgen(n::Int, ordered = false)
```

If `ordered` we start with high `|cov[i,j]|` as `|i-j| = 1` and decreases `|cov[i,j]|` as `|i-j|` rises. Else covariance matrix elements are more random.

```julia
julia> srand(43);

julia> cormatgen(5)
5×5 Array{Float64,2}:
  1.0       -0.464674   0.352662   0.452469   0.273749
 -0.464674   1.0       -0.143304  -0.589167  -0.354571
  0.352662  -0.143304   1.0        0.643956   0.628471
  0.452469  -0.589167   0.643956   1.0        0.356678
  0.273749  -0.354571   0.628471   0.356678   1.0
```

```julia
julia> srand(43);

julia> cormatgen(5, true)
5×5 Array{Float64,2}:
  1.0       -0.911958  -0.867202   0.84641    0.623197
 -0.911958   1.0        0.928058  -0.924794  -0.681956
 -0.867202   0.928058   1.0       -0.936033  -0.769087
  0.84641   -0.924794  -0.936033   1.0        0.743044
  0.623197  -0.681956  -0.769087   0.743044   1.0

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

To generate `t` realisations of `n`-variate data from Clayton copula with paramete `\theta = 1.` run

```julia
julia> claytoncopulagen(t::Int, n::Int = 2)
```

```julia
julia> srand(43);

julia> claytoncopulagen(10, 2)
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

### Converting marinals

To convert marginals of `U \in [0,1]^n` using one type univariate of distributions `dist` with parameters `p[i]` for `i` th marginal run:

```julia
julia> convertmarg!(U::Matrix{T}, dist::Distribution, p::Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}; testunif::Bool = true)
```

It `testunif` each marginal is tested for uniformity.

```julia
julia> srand(43);

julia> x = rand(10,2);

julia> convertmarg!(x, Normal, [[0, 1],[0, 1]])

julia> x
10×2 Array{Float64,2}:
 -0.911655    4.17328
  0.756673  -14.4472 
  1.22088   -11.4823 
  1.43866   -13.1053 
 -0.231978  -11.2415 
  1.35696     6.43914
  0.949145  -26.0172 
 -0.251957  -18.9723 
 -0.177808    0.54172
  1.70477    10.4192 
```

# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.
