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
decreasing generation function parametrised by `θ`, such that `φ(t): [0,1] →
[0, ∞)` and `φ⁻¹(s)` is the pseudo-inverse. In `n`-variate case
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

 * If `cor = kendall`, uses Kendall's τ correlation coefficients `θ`.
 * If `cor = pearson`, uses Pearson ρ correlation coefficient instead of `θ`.
 * If `reversed = true` returns data from reversed copula.

To generate data from reversed copula:

 * Generated data from corresponding copula `[u₁, ..., uᵢ, ..., uₙ]`,
 * Perform  transformation  `∀ᵢ uᵢ → 1-uᵢ`.

For modelling justification see: K. Domino, T. Błachowicz, M. Ciupak, 'The use of copula functions for predictive analysis of correlations between extreme storm tides',
Physica A: Statistical Mechanics and its Applications 413, 489-497, 2014, and K. Domino, T. Błachowicz, 'The use of copula functions for modeling the risk of
investment in shares traded on the Warsaw Stock Exchange', Physica A: Statistical Mechanics and its Applications 413, 77-85, 2014.


### Nested archimedean copulas

To generate `t` realisations of `∑ᵢ nᵢ + m` variate data from nested
archimedean copulas,  McNeil, A.J., 2008. 'Sampling nested Archimedean
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

julia> srand(43);

julia> nestedarchcopulagen(10, [2,2], [2., 3.], 1.1, "clayton", 1)
10×5 Array{Float64,2}:
 0.414567  0.683167   0.9953    0.607738  0.793386
 0.533001  0.190563   0.17076   0.273119  0.78807
 0.572782  0.161307   0.418821  0.110356  0.661781
 0.623807  0.140974   0.295422  0.454368  0.477065
 0.386276  0.266261   0.559423  0.449874  0.294137
 0.219757  0.122586   0.371318  0.298965  0.507315
 0.322658  0.0627113  0.738565  0.919912  0.19471
 0.131938  0.0672061  0.364721  0.220329  0.662842
 0.773414  0.812113   0.639333  0.527118  0.545043
 0.958656  0.871822   0.958339  0.801866  0.862751

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

julia> srand(43)

julia> x = nestedarchcopulagen(5, [4., 3., 2.], "gumbel")

5×4 Array{Float64,2}:
 0.483466  0.621572  0.241025  0.312664
 0.827237  0.696634  0.768802  0.730543
 0.401159  0.462126  0.412573  0.72571
 0.970726  0.964746  0.940314  0.934625
 0.684486  0.614142  0.690664  0.401897
```

To generate `t` realisations of `∑ᵢ ∑ⱼ nᵢⱼ` variate data from double nested gumbel copula:
`C_θ(C_ϕ₁(C_Ψ₁₁(u,...), ..., C_C_Ψ₁,ₗ₁(u...)), ..., C_ϕₖ(C_Ψₖ₁(u,...), ..., C_Ψₖ,ₗₖ(u,...)))`
 where `lᵢ = length(n[i])` run:

```julia

julia> nestedarchcopulagen::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}}, ϕ::Vector{Float64}, θ₀::Float64, copula::String = "gumbel")

```

```julia
julia> srand(43)

julia> x = nestedarchcopulagen(5, [[2,2],[2]], [[3., 2.], [4.]], [1.5, 2.1], 1.2, "gumbel")
5×6 Array{Float64,2}:
0.464403  0.711722   0.883035   0.896706   0.888614   0.826514
0.750596  0.768193   0.0659561  0.0252472  0.996014   0.989127
0.825211  0.712079   0.581356   0.507739   0.882675   0.84959
0.276326  0.0827071  0.240836   0.434629   0.0184611  0.031363
0.208422  0.504727   0.27561    0.639089   0.481855   0.573715

```

### Mixture of bivariate archimedean sub-copulas


To generate `t` realisations of `length(θ)+1` using one parameter bivariate archimedean sub-copulas from the same familly, with parameter `θᵢ` for each
neighbour marginals (i'th and i+1'th)  run:

```julia

julia> bivariatecopgen(t::Int, θ::Union{Vector{Float64}, Vector{Int}}, copula::String; rev::Bool = false, cor::String = "")

```

In other words `∀i∈[1, length(θ)]` data are generated form the following archimedean copula `C_{θᵢ}(uᵢ, u_{i+1})`. Due to features of bivariate copulas, each marginal `uᵢ` is uniformly
distributed on `[0,1]`, hence we got a multivariate copula, defined by
subsequent bivariate sub-copulas. The relation between marginals `i` and `j`: `i ≠ j+1` are defined by a sequence of
bivariate copulas.

Following families are supported: "clayton", "frank" and
"amh" -  Ali-Mikhail-Haq. Conditions for `θᵢ` parameters are such as in
bivariate copulas cases. Dislike the nested copula example, ordering of `θ` elements do not matter.
Reversed copula and the use of correlations instead of copula parameter are also
supported.

```julia

julia> srand(43);

julia> bivariatecopgen(10, [4., 11.], "frank")
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

To generate `t` realisations of `n`-variate data from Marshall-Olkin copula with parameter series `λ` of of non-negative elements `λₛ`, run:

```julia
julia> marshalolkincopulagen(t::Int, λ::Vector{Float64}; reverse::Bool = false)
```

Number of marginals is `n = ceil(Int, log(2, length(λ)-1))`.
Parameters are ordered as follow: `λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]`
If `reversed = true`, returns data from reversed Marshal-Olkin copula , i.e. generates data `[u₁, ..., uᵢ, ..., uₙ]` from given Marshall-Olkin copula and perform transformation `∀ᵢ uᵢ → 1-uᵢ`


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

To generate data from the  Marshall-Olkin copula we use algorithm presented P. Embrechts, F. Lindskog, A McNeil 'Modelling Dependence with Copulas and Applications to Risk Management', 2001
∗∗


## Frechet family copulas

To generate `t` realisation of `n` variate one parameter Frechet copula `Cf = α C_{max} + (1-α) C_{⟂}`, where `0 ≤ α ≤ 1` run:


```julia
julia> frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64})
```

```julia

julia> srand(43);

julia> frechetcopulagen(10, 2, 0.5)
10×2 Array{Float64,2}:
 0.180975  0.180975
 0.775377  0.0742681
 0.888934  0.888934
 0.924876  0.0950087
 0.408278  0.408278
 0.912603  0.740184
 0.828727  0.828727
 0.400537  0.0288987
 0.429437  0.429437
 0.955881  0.851275
```

Two parameters Frechet copula, `C = α C_{max} + β C_{min} + (1- α - β) C_{⟂}`
is supported only for `n == 2`:

```
julia> frechetcopulagen(t::Int, n::Int, α::Union{Int, Float64}, β::Union{Int, Float64})
```

Here where `0 ≤ α` , where `0 ≤ β` and `α + β ≤ 1`

``` julia

julia> srand(43);

julia> frechetcopulagen(10, 2, 0.3, 0.2)
10×2 Array{Float64,2}:
 0.661781    0.661781  
 0.0742681   0.775377  
 0.125437    0.874563  
 0.0950087   0.924876  
 0.130474    0.130474  
 0.740184    0.912603  
 0.00463791  0.00463791
 0.0288987   0.400537  
 0.521601    0.478399  
 0.851275    0.955881

 ```

### Mixture of bivariate Frechet sub-copulas


To generate `t` realisations of `length(α)+1` multivariate  data using two parameter bivariate Frechet sub-copulas with parameter `αᵢ` and
`βᵢ` for each neighbour  (i'th and i+1'th) marginals  run:


```julia

julia> bivfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))

```
In other words `∀i∈[1, length(θ)]` data are generated from following Frechet copula `C_{αᵢ,βᵢ}(uᵢ, u_{i+1})`. Due to features of bivariate copulas, each marginal `uᵢ` is uniformly
distributed on `[0,1]`, hence we got a multivariate copula, defined by subsequent bivariate sub-copulas.
The relation between marginals `i` and `j`: `i ≠ j+1` are defined by a sequence of
bivariate copulas. For some parameters `α,β` such copula would be equivalent to
a hierarchically nested copula.


```julia
julia> srand(43)

julia> julia> bivfrechetcopulagen(10, [0.6, 0.4], [0.3, 0.5])
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


## Helpers

### Correlation matrix generation

To generate a `n x n` correlation matrix `Σ`, with reference correlation `0 > ρ > 1` run:

```julia
julia> cormatgen(n::Int, ρ::Float64 = 0.5, ordered::Bool = false, altersing::Bool = true)
```

 * If `ordered = false` `Σ` elements varies around `ρ`, i.e. `σᵢⱼ ≈ ρ+δ`, else they drop
as indices differences rise, i.e. `σᵢⱼ ≳ σᵢₖ` as `|i-j| < |i-k|`.
 * If `altersing = true`, some `σ` are positive and some negative, else `∀ᵢⱼ σᵢⱼ ≥ 0`.

```julia
julia> srand(43);

julia> cormatgen(4)
4×4 Array{Float64,2}:
  1.0        0.574741  -0.789649  -0.654538
  0.574741   1.0       -0.717196  -0.610049
 -0.789649  -0.717196   1.0        0.703387
 -0.654538  -0.610049   0.703387   1.0
```
```

```julia
julia> srand(43);

julia> cormatgen(4, 0.5, true)
4×4 Array{Float64,2}:
  1.0        -0.417562  -0.435673  -0.0977413
 -0.417562    1.0        0.703997   0.389709 
 -0.435673    0.703997   1.0        0.536093 
 -0.0977413   0.389709   0.536093   1.0  

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

## Copula mixture

The following function would allow us to generate `t > 2` realisations of `n` variate data with
marginals numerates by `i` from Gaussian copula with given correlation matrix
`Σ` and replace
some sub-copulas (with marginals numerates by `j₁, j₂, ..., jₛ: jₖ = [jₖ₁, ..., jₖₙₖ]`) by non-Gaussian one. The function returns `[0,1]ⁿ ∋  u: ∀ᵢ uᵢ ∼ Uniform(0,1)` such that the overall correlation of `u`
is similar to the overall correlation given by correlation matrix `Σ`, i.e. `||Σ|| ≈ ||cor(quantile(Normal(0,1), u))||`, where `||.||` is a vector norm.

Following conditions on multi-indices are required `∀ₖₗ jₖ ∩ jₗ = ∅` and `∀ₖ jₖ ⊆ i`
and `∑ₖ length(jₖ) ≤ n + δ` where `δ` is an integer such as `δ = 0, 1` and
depends no sub-copulas chosen. For chosen `jₖ` we have a k'th non-Gaussian sub copula, for multi-index `g = i \ ∪ ₖ jₖ`
we have Gaussian sub-copula. If we chose a multi-index which parts belong to different `jₖ` or different `jₖ` and `g`
the sub-copula is more complex.

Families of sub-copual and corresponding marginals `jₖ` are supplied
in the form of the Vector{Pair{String,Vector{Int64}}}, following families are supported:

* Archimedean: nested (or bivariate) "gumbel", "frank", "clayton" or/and "amh" copula
* "t-student"
* "frechet" - mixture of bivariate frechet copulas,
*  "mo" - "Marshal-Olkin" copula is supported only for `length(jₖ) = 2 or 3`.

To generate data run:

``` julia

julia> copulamix(t::Int, Σ::Matrix{Float}, inds::Vector{Pair{String,Vector{Int64}}}; λ::Vector{Float} = [6., 3., 1., 15.], ν::Int = 2, a::Vector{Float} = [0.1])

```

The function takes also following named parameters:

* λ - vector of chosen parameters of "Marshal-Olkin" copula, in bivariate case
[λ₁ λ₂] - by default = [6., 3.], in tri-variate case [λ₁ λ₂ λ₃, λ₁₂₃] - by
default = [6., 3., 1., 15.], reminding parameters λᵢⱼ
are calculated from required Pearson correlation;
 * ν - is a parameter of t-student copula, the number of degrees of freedom,
* a - is a difference of frechet bivariate copulas parameters, a = α - β.

For exemplary use see:

```

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

# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.
