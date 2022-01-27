"""
    simulate_copula(t::Int, copula::Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the Gumbel copula -  Gumbel_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(2, Gumbel_cop(3, 1.5))
2×3 Array{Real,2}:
 0.740038  0.918928  0.950674
 0.637826  0.483514  0.123949
```
"""
function simulate_copula(t, copula::Gumbel_cop{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Clayton_cop; rng::AbstractRNG = Random.GLOBAL_RNG)


Returns t realizations from the Clayton copula - Clayton_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(3, Clayton_cop(2, 1.))
3×2 Array{Float64,2}:
 0.562482  0.896247
 0.968953  0.731239
 0.749178  0.38015

 julia> Random.seed!(43);

 julia> simulate_copula(2, Clayton_cop(2, -.5))
 2×2 Array{Float64,2}:
  0.180975  0.818017
  0.888934  0.863358
```
"""
function simulate_copula(t, copula::Clayton_cop{T}; rng= Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end


"""
    simulate_copula(t::Int, copula::Gumbel_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the Gumbel _cop _rev(n, θ) - the reversed Gumbel copula (reversed means u → 1 .- u).

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(2, Gumbel_cop_rev(3, 1.5))
2×3 Array{Real,2}:
 0.259962  0.081072  0.0493259
 0.362174  0.516486  0.876051
```
"""
function simulate_copula(t, copula::Gumbel_cop_rev{T}; rng= Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Clayton_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations form the Clayton _cop _rev(n, θ) - the reversed Clayton copula (reversed means u → 1 .- u)

```jldoctest

  julia> Random.seed!(43);

 julia> simulate_copula(2, Clayton_cop_rev(2, -0.5))
 2×2 Array{Float64,2}:
   0.819025  0.181983
   0.111066  0.136642
```
"""
function simulate_copula(t, copula::Clayton_cop_rev{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::AMH_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the Ali-Mikhail-Haq copula- AMH_cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(4, AMH_cop(2, 0.5))
4×2 Array{Float64,2}:
 0.483939  0.883911
 0.962064  0.665769
 0.707543  0.25042
 0.915491  0.494523

julia> Random.seed!(43);

julia> simulate_copula(4, AMH_cop(2, -0.5))
4×2 Array{Float64,2}:
 0.180975  0.820073
 0.888934  0.886169
 0.408278  0.919572
 0.828727  0.335864
```
"""
function simulate_copula(t, copula::AMH_cop{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::AMH_cop_rev; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the reversed Ali-Mikhail-Haq copula - AMH _cop _rev(n, θ), reversed means u → 1 .- u.

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(4, AMH_cop_rev(2, 0.5))
4×2 Array{Float64,2}:
 0.516061   0.116089
 0.0379356  0.334231
 0.292457   0.74958
 0.0845089  0.505477


julia> simulate_copula(4, AMH_cop_rev(2, -0.5))
4×2 Array{Float64,2}:
 0.819025  0.179927
 0.111066  0.113831
 0.591722  0.0804284
 0.171273  0.664136
```
"""
function simulate_copula(t, copula::AMH_cop_rev{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T ,t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Frank_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations from the n-variate Frank copula - Frank _cop(n, θ)

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(4, Frank_cop(2, 3.5))
4×2 Array{Float64,2}:
 0.650276  0.910212
 0.973726  0.789701
 0.690966  0.358523
 0.747862  0.29333

julia> Random.seed!(43);

julia> simulate_copula(4, Frank_cop(2, 0.2, SpearmanCorrelation))
4×2 Array{Float64,2}:
 0.504123  0.887296
 0.962936  0.678791
 0.718628  0.271543
 0.917759  0.51439
```
"""
function simulate_copula(t, copula::Frank_cop{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Chain_of_Archimedeans; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of multivariate data modeled by the chain of bivariate
Archimedean copulas, i.e.

    Chain_of_Archimedeans(θ::Vector{Flota64}, copulas::Union{String, Vector{String}})

```jldoctest
julia> Random.seed!(43);

julia> c = Chain_of_Archimedeans([4., 11.], "frank")
Chain_of_Archimedeans(3, [4.0, 11.0], ["frank", "frank"])

julia> simulate_copula(1, c)
1×3 Array{Float64,2}:
 0.180975  0.492923  0.679345

julia> c = Chain_of_Archimedeans([.5, .7], ["frank", "clayton"], KendallCorrelation)
Chain_of_Archimedeans(3, [5.736282707019972, 4.666666666666666], ["frank", "clayton"])

julia> Random.seed!(43);

julia> simulate_copula(1, c)
1×3 Array{Float64,2}:
 0.180975  0.408582  0.646887
```
"""
function simulate_copula(t, copula::Chain_of_Archimedeans{T}; rng= Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Chain_of_Frechet; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations modeled by the chain of bivariate two parameter Frechet copulas

```jldoctest
julia> Random.seed!(43)

julia> simulate_copula(10, Chain_of_Frechet([0.6, 0.4], [0.3, 0.5]))
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
"""
function simulate_copula(t, copula::Chain_of_Frechet{T}; rng = Random.GLOBAL_RNG) where T
  U = zeros(T, t, copula.n)
  simulate_copula!(U, copula; rng = rng)
  return U
end

"""
    simulate_copula(t::Int, copula::Gaussian_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of the Gaussian copula

    Gaussian_cop(Σ)

```jldoctest

julia> Random.seed!(43);

julia> simulate_copula(10, Gaussian_cop([1. 0.5; 0.5 1.]))
10×2 Array{Real,2}:
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
"""
function simulate_copula(t, copula::Gaussian_cop{T}; rng = Random.GLOBAL_RNG) where T
  U = zeros(T, t, copula.n)
  simulate_copula!(U, copula; rng = rng)
  return U
end
"""
    simulate_copula(t::Int, copula::Student _cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of the t-Student Copula

    Student_cop(Σ, ν)

where: Σ - correlation matrix, ν - degrees of freedom.

```jldoctest
julia> Random.seed!(43);

julia> simulate_copula(10, Student_cop([1. 0.5; 0.5 1.], 10))
10×2 Array{Real,2}:
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
"""
function simulate_copula(t, copula::Student_cop{T}; rng = Random.GLOBAL_RNG) where T
  U = zeros(T, t, copula.n)
  simulate_copula!(U, copula; rng = rng)
  return U
end

"""
    simulate_copula(t::Int, copula::Frechet_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizatioins of data from the Frechet copula

    Frechet_cop(n, α)
    Frechet_cop(n, α, β)

```jldoctest

julia> Random.seed!(43);

julia> f = Frechet_cop(3, 0.5)
Frechet_cop(3, 0.5, 0.0)

julia> simulate_copula(1, f)
1×3 Array{Real,2}:
0.180975  0.775377  0.888934

julia> Random.seed!(43);

julia> f = Frechet_cop(2, 0.5, 0.2)
Frechet_cop(2, 0.5, 0.2)

julia> simulate_copula(1, f)
1×2 Array{Real,2}:
0.180975  0.775377
```
"""

function simulate_copula(t, copula::Frechet_cop{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Marshall_Olkin_cop(λ); rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of the n-variate Marshall-Olkin copula:

    Marshall_Olkin_cop(λ)

```jldoctest

julia> Random.seed!(43);

julia> cop = Marshall_Olkin_cop([1.,2.,3.])
Marshall_Olkin_cop(2, [1.0, 2.0, 3.0])

julia> simulate_copula(1, cop)
1×2 Array{Float64,2}:
  0.854724  0.821831
```
"""
function simulate_copula(t, copula::Marshall_Olkin_cop{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Hierarchical_Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of multivariate data from hierarchically nested Gumbel copula, i.e.

    Hierarchical_Gumbel_cop(θ)

```jldoctest
julia> using Random

julia> Random.seed!(43);

julia> c = Hierarchical_Gumbel_cop([5., 4., 3.])
Hierarchical_Gumbel_cop(4, [5.0, 4.0, 3.0])

julia> simulate_copula(3, c)
3×4 Array{Float64,2}:
 0.100353  0.207903  0.0988337  0.0431565
 0.347417  0.217052  0.223734   0.042903
 0.73617   0.347349  0.168348   0.410963
```
"""
function simulate_copula(t, copula::Hierarchical_Gumbel_cop{T}; rng = Random.GLOBAL_RNG) where T
    U = zeros(T, t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end

"""
    simulate_copula(t::Int, copula::Nested_Clayton_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of data generated using Nested Clayton copula

```jldoctest

julia> Random.seed!(43);

julia> c1 = Clayton_cop(2, 2.)
Clayton_cop(2, 2.0)

julia> c2 = Clayton_cop(2, 3.)
Clayton_cop(2, 3.0)

julia> cp = Nested_Clayton_cop([c1, c2], 1, 1.1)
Nested_Clayton_cop(Clayton_cop[Clayton_cop(2, 2.0), Clayton_cop(2, 3.0)], 1, 1.1)

julia> simulate_copula(4, cp)
4×5 Array{Float64,2}:
0.514118  0.84089   0.870106  0.906233  0.739349
0.588245  0.85816   0.935308  0.944444  0.709009
0.59625   0.665947  0.483649  0.603074  0.153501
0.200051  0.304099  0.242572  0.177836  0.0851603
```
"""
function simulate_copula(t, copula::Nested_Clayton_cop{T}; rng = Random.GLOBAL_RNG) where T
     U = zeros(T, t, copula.n)
     simulate_copula!(U, copula; rng = rng)
     return U
end


"""
     simulate_copula(t::Int, copula::Nested_AMH_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of data generated using Nested AMH copula

```jldoctest

julia> c1 = AMH_cop(2, .7)
AMH_cop(2, 0.7)

julia> c2 = AMH_cop(2, .8)
AMH_cop(2, 0.8)

julia> cp = Nested_AMH_cop([c1, c2], 1, 0.2)
Nested_AMH_cop(AMH_cop[AMH_cop(2, 0.7), AMH_cop(2, 0.8)], 1, 0.2)

julia> Random.seed!(43);

julia> simulate_copula(4, cp)
4×5 Array{Float64,2}:
 0.557393  0.902767  0.909853  0.938522  0.586068
 0.184204  0.866664  0.699134  0.226744  0.102932
 0.268634  0.383355  0.179023  0.533749  0.995958
 0.578143  0.840169  0.743728  0.963226  0.576695
```
"""
function simulate_copula(t, copula::Nested_AMH_cop{T}; rng = Random.GLOBAL_RNG) where T
     U = zeros(T, t, copula.n)
     simulate_copula!(U, copula; rng = rng)
     return U
end


"""
    simulate_copula(t::Int, copula::Nested_Frank_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of data generated using Nested Frank copula
```jldoctest

julia> c1 = Frank_cop(2, 4.)
Frank_cop(2, 4.0)

julia> c2 = Frank_cop(2, 5.)
Frank_cop(2, 5.0)

julia> c = Nested_Frank_cop([c1, c2],1, 2.0)
Nested_Frank_cop(Frank_cop[Frank_cop(2, 4.0), Frank_cop(2, 5.0)], 1, 2.0)

julia> Random.seed!(43);

julia> simulate_copula(1, c)
1×5 Array{Float64,2}:
 0.642765  0.901183  0.969422  0.9792  0.74155
```
"""
function simulate_copula(t, copula::Nested_Frank_cop{T}; rng = Random.GLOBAL_RNG) where T
     U = zeros(T, t, copula.n)
     simulate_copula!(U, copula; rng = rng)
     return U
end

"""
    simulate_copula(t::Int, copula::Nested_Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Returns t realizations of data generated using  Nested Gumbel copula

```jldoctest

julia> Random.seed!(43);

julia> c1 = Gumbel_cop(2, 2.)
Gumbel_cop(2, 2.0)

julia> c2 = Gumbel_cop(2, 3.)
Gumbel_cop(2, 3.0)

julia> cp = Nested_Gumbel_cop([c1, c2], 1, 1.1)
Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 2.0), Gumbel_cop(2, 3.0)], 1, 1.1)

julia> simulate_copula(4, cp)
4×5 Array{Float64,2}:
 0.387085   0.693399   0.94718   0.953776  0.583379
 0.0646972  0.0865914  0.990691  0.991127  0.718803
 0.966896   0.709233   0.788019  0.855622  0.755476
 0.272487   0.106996   0.756052  0.834068  0.661432
```
"""
function simulate_copula(t, copula::Nested_Gumbel_cop{T}; rng = Random.GLOBAL_RNG) where T
     U = zeros(T, t, copula.n)
     simulate_copula!(U, copula; rng = rng)
     return U
end

"""
    simulate_copula(t::Int, copula::Double_Nested_Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)

Simulate t realization of the Double Nested Gumbel copula i.e.

    Double_Nested_Gumbel_cop(vec_of_children, θ)

```jldoctest
julia> a = Gumbel_cop(2, 5.)
Gumbel_cop(2, 5.0)

julia> b = Gumbel_cop(2, 6.)
Gumbel_cop(2, 6.0)

julia> c = Gumbel_cop(2, 5.5)
Gumbel_cop(2, 5.5)

julia> p1 = Nested_Gumbel_cop([a,b], 1, 2.)
Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.0), Gumbel_cop(2, 6.0)], 1, 2.0)

julia> p2 = Nested_Gumbel_cop([c], 2, 2.1)
Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.5)], 2, 2.1)

julia> copula = Double_Nested_Gumbel_cop([p1, p2], 1.5)
Double_Nested_Gumbel_cop(Nested_Gumbel_cop[Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.0), Gumbel_cop(2, 6.0)], 1, 2.0), Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.5)], 2, 2.1)], 1.5)

julia> Random.seed!(43);

julia> simulate_copula(5, copula)
5×9 Array{Float64,2}:
 0.598555   0.671584   0.8403     0.846844  0.634609  0.686927  0.693906  0.651968    0.670812
 0.0518892  0.191236   0.0803859  0.104325  0.410727  0.529354  0.557387  0.370518    0.592302
 0.367914   0.276196   0.382616   0.470171  0.264135  0.144503  0.13097   0.00687015  0.01417
 0.632727   0.596879   0.244176   0.338809  0.58771   0.147539  0.219099  0.287937    0.0569943
 0.310365   0.0483216  0.119312   0.107155  0.336619  0.279602  0.262756  0.438432    0.403061
```
"""
function simulate_copula(t, copula::Double_Nested_Gumbel_cop{T}; rng = Random.GLOBAL_RNG) where T
     U = zeros(T, t, copula.n)
     simulate_copula!(U, copula; rng = rng)
     return U
end