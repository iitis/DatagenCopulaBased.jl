# nested archimedean copulas

# Algorithms from:
# M. Hofert, `Efficiently sampling nested Archimedean copulas` Computational Statistics and Data Analysis 55 (2011) 57–70
# M. Hofert, 'Sampling  Archimedean copulas', Computational Statistics & Data Analysis, Volume 52, 2008
# McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'. Journal of Statistical Computation and Simulation 78, 567–581.

#Basically we use Alg. 5 of McNeil, A.J., 2008. 'Sampling nested Archimedean copulas'.

"""
    Nested_Clayton_cop

Fields:
- children::Vector{Clayton_cop}  vector of children copulas
- m::Int ≧ 0 - number of additional marginals modeled by the parent copula only
- θ::Float64 - parameter of parent copula, domain θ > 0.

Nested Clayton copula: C_θ(C_ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C_ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, the last m variables will be modeled by the parent copula only.

Constructor

    Nested_Clayton_cop(children::Vector{Clayton_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting condition requires
θ <= minimum(ϕ)

Constructor

    Nested_Clayton_cop(children::Vector{Clayton_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ.

```jldoctest
julia> a = Clayton_cop(2, 2.)
Clayton_cop(2, 2.0)

julia> Nested_Clayton_cop([a], 2, 0.5)
Nested_Clayton_cop(Clayton_cop[Clayton_cop(2, 2.0)], 2, 0.5)

julia> Nested_Clayton_cop([a, a], 2, 0.5)
Nested_Clayton_cop(Clayton_cop[Clayton_cop(2, 2.0), Clayton_cop(2, 2.0)], 2, 0.5)

```
"""
struct Nested_Clayton_cop
  children::Vector{Clayton_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_Clayton_cop})(children::Vector{Clayton_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "clayton")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      maximum(ϕ) < θ+2*θ^2+750*θ^5 || @warn("θ << ϕ, marginals may not be uniform")
      new(children, m, θ)
  end
  function(::Type{Nested_Clayton_cop})(children::Vector{Clayton_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "clayton", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      maximum(ϕ) < θ+2*θ^2+750*θ^5 || @warn("θ << ϕ, marginals may not be uniform")
      new(children, m, θ)
  end
end

"""
    Nested_AMH_cop

Nested Ali-Mikhail-Haq copula, fields:
- children::Vector{AMH _cop}  vector of children copulas
- m::Int ≧ 0 - number of additional marginals modeled by the parent copula only
- θ::Float64 - parameter of parent copula, domain θ ∈ (0,1).

Nested Ali-Mikhail-Haq copula: C _θ(C _ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C _ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, the last m variables will be modeled by the parent copula only.

Constructor

    Nested_AMH_cop(children::Vector{AMH_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting condition requires
θ <= minimum(ϕ)

Constructor

    Nested_AMH_cop(children::Vector{AMH_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ.
```jldoctest

julia> a = AMH_cop(2, .2)
AMH_cop(2, 0.2)

julia> Nested_AMH_cop([a, a], 2, 0.1)
Nested_AMH_cop(AMH_cop[AMH_cop(2, 0.2), AMH_cop(2, 0.2)], 2, 0.1)

```
"""
struct Nested_AMH_cop
  children::Vector{AMH_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_AMH_cop})(children::Vector{AMH_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "amh")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, m, θ)
  end
  function(::Type{Nested_AMH_cop})(children::Vector{AMH_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "amh", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, m, θ)
  end
end

"""
    Nested_Frank_cop

Fields:
- children::Vector{Frank_cop}  vector of children copulas
- m::Int ≧ 0 - number of additional marginals modeled by the parent copula only
- θ::Float64 - parameter of parent copula, domain θ ∈ (0,∞).

Nested Frank copula: C _θ(C _ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C _ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, the last m variables will be modeled by the parent copula only.

Constructor

    Nested_Frank_cop(children::Vector{Frank_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting condition requires
θ <= minimum(ϕ)

Constructor

    Nested_Frank_cop(children::Vector{Frank_ cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ.

```jldoctests

julia> a = Frank_cop(2, 5.)
Frank_cop(2, 5.0)

julia> Nested_Frank_cop([a, a], 2, 0.1)
Nested_Frank_cop(Frank_cop[Frank_cop(2, 5.0), Frank_cop(2, 5.0)], 2, 0.1)
```
"""
struct Nested_Frank_cop
  children::Vector{Frank_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_Frank_cop})(children::Vector{Frank_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "frank")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, m, θ)
  end
  function(::Type{Nested_Frank_cop})(children::Vector{Frank_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "frank", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, m, θ)
  end
end

"""
    Nested_Gumbel_cop

Fields:
- children::Vector{Gumbel_cop}  vector of children copulas
- m::Int ≧ 0 - number of additional marginals modeled by the parent copula only
- θ::Float64 - parameter of parent copula, domain θ ∈ [1,∞).

Nested Gumbel copula: C _θ(C _ϕ₁(u₁₁, ..., u₁,ₙ₁), ..., C _ϕₖ(uₖ₁, ..., uₖ,ₙₖ), u₁ , ... uₘ).
If m > 0, the last m variables will be modeled by the parent copula only.

Constructor

    Nested_Gumbel_cop(children::Vector{Gumbel_cop}, m::Int, θ::Float64)

Let ϕ be the vector of parameter of children copula, sufficient nesting condition requires
θ <= minimum(ϕ)

Constructor

    Nested_Gumbel_cop(children::Vector{Gumbel_cop}, m::Int, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ.
```jldoctest

julia> a = Gumbel_cop(2, 5.)
Gumbel_cop(2, 5.0)

julia> Nested_Gumbel_cop([a, a], 2, 2.1)
Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.0), Gumbel_cop(2, 5.0)], 2, 2.1)
```
"""
struct Nested_Gumbel_cop
  children::Vector{Gumbel_cop}
  m::Int
  θ::Float64
  function(::Type{Nested_Gumbel_cop})(children::Vector{Gumbel_cop}, m::Int, θ::Float64)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      testθ(θ, "gumbel")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, m, θ)
  end
  function(::Type{Nested_Gumbel_cop})(children::Vector{Gumbel_cop}, m::Int, ρ::Float64, cor::String)
      m >= 0 || throw(DomainError("not supported for m  < 0 "))
      θ = getθ4arch(ρ, "gumbel", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, m, θ)
  end
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
function simulate_copula(t::Int, copula::Nested_Frank_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
    m = copula.m
    θ = copula.θ
    children = copula.children
    ϕ = [ch.θ for ch in children]
    n = [ch.n for ch in children]

    ws = [logseriescdf(1-exp(theta)) for theta in ϕ]
    n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
    n2 = sum(n)+m
    w = logseriescdf(1-exp(-θ))
    U = zeros(t, n2)
    for j in 1:t
       rand_vec = rand(rng, n2+1)
       U[j,:] = nested_frank_gen(n1, ϕ, θ, rand_vec, w, ws; rng=rng)
   end
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
function simulate_copula(t::Int, copula::Nested_AMH_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
     m = copula.m
     θ = copula.θ
     children = copula.children
     ϕ = [ch.θ for ch in children]
     n = [ch.n for ch in children]
     n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
     n2 = sum(n)+m
     U = zeros(t, n2)
     for j in 1:t
        rand_vec = rand(rng, n2+1)
        U[j,:] = nested_amh_gen(n1, ϕ, θ, rand_vec; rng=rng)
    end
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
function simulate_copula(t::Int, copula::Nested_Clayton_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
     m = copula.m
     θ = copula.θ
     children = copula.children
     ϕ = [ch.θ for ch in children]
     n = [ch.n for ch in children]
     n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
     n2 = sum(n)+m
     U = zeros(t, n2)
     for j in 1:t
        rand_vec = rand(rng, n2+1)
        U[j,:] = nested_clayton_gen(n1, ϕ, θ, rand_vec; rng=rng)
    end
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

julia> simulate_copula(4, cp)
Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 2.0), Gumbel_cop(2, 3.0)], 1, 1.1)

julia> simulate_copula(4, cp)
4×5 Array{Float64,2}:
 0.387085   0.693399   0.94718   0.953776  0.583379
 0.0646972  0.0865914  0.990691  0.991127  0.718803
 0.966896   0.709233   0.788019  0.855622  0.755476
 0.272487   0.106996   0.756052  0.834068  0.661432

```
"""
function simulate_copula(t::Int, copula::Nested_Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
     m = copula.m
     θ = copula.θ
     children = copula.children
     ϕ = [ch.θ for ch in children]
     n = [ch.n for ch in children]
     n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
     n2 = sum(n)+m

     U = zeros(t, n2)
     for j in 1:t
        rand_vec = rand(rng, n2)
        U[j,:] = nested_gumbel_gen(n1, ϕ, θ, rand_vec; rng=rng)
    end
return U
end

"""
    Double_Nested_Gumbel_cop

Fields:
- children::Vector{Nested _Gumbel _cop}  vector of children copulas
- θ::Float64 - parameter of parent copula, domain θ ∈ [1,∞).

Constructor

    Double_Nested_Gumbel _cop(children::Vector{Nested_Gumbel_cop}, θ::Float64)
requires sufficient nesting condition for θ and child copulas.

Constructor

    Doulbe_Nested_Gumbel_cop(children::Vector{Nested_Gumbel_cop}, θ::Float64, cor::String)
uses "Spearman" or "Kendall" correlation to compute θ.

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

julia> Double_Nested_Gumbel_cop([p1, p2], 1.5)
Double_Nested_Gumbel_cop(Nested_Gumbel_cop[Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.0), Gumbel_cop(2, 6.0)], 1, 2.0), Nested_Gumbel_cop(Gumbel_cop[Gumbel_cop(2, 5.5)], 2, 2.1)], 1.5)
```
"""
struct Double_Nested_Gumbel_cop
  children::Vector{Nested_Gumbel_cop}
  θ::Float64
  function(::Type{Double_Nested_Gumbel_cop})(children::Vector{Nested_Gumbel_cop}, θ::Float64)
      testθ(θ, "gumbel")
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, θ)
  end
  function(::Type{Double_Nested_Gumbel_cop})(children::Vector{Nested_Gumbel_cop}, ρ::Float64, cor::String)
      θ = getθ4arch(ρ, "gumbel", cor)
      ϕ = [ch.θ for ch in children]
      θ <= minimum(ϕ) || throw(DomainError("violated sufficient nesting condition"))
      new(children, θ)
  end
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
function simulate_copula(t::Int, copula::Double_Nested_Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
    θ = copula.θ
    v = copula.children
    ns = [[ch.n for ch in vs.children] for vs in v]
    Ψs = [[ch.θ for ch in vs.children] for vs in v]
    dims = sum([sum(ns[i])+v[i].m for i in 1:length(v)])
    U = zeros(t, dims)
    for j in 1:t
        X = Float64[]
        for k in 1:length(v)
            n = ns[k]
            n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
            n2 = sum(n)+v[k].m
            rand_vec = rand(rng, n2)
            X = vcat(X, nested_gumbel_gen(n1, Ψs[k], v[k].θ./θ, rand_vec; rng = rng))
        end
        X = -log.(X)./levyel(θ, rand(rng), rand(rng))
        U[j,:] = exp.(-X.^(1/θ))
    end
    return U
end

#=
"""
    nested_gumbel(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)

Sample nested Gumbel copula, axiliary function for simulate_copula(t::Int, copula::Double_Nested_Gumbel_cop)
"""
function nested_gumbel(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, m::Int = 0)
  n1 = vcat([collect(1:n[1])], [collect(cumsum(n)[i]+1:cumsum(n)[i+1]) for i in 1:length(n)-1])
  n2 = sum(n)+m
  return nestedcopulag("gumbel", n1, ϕ, θ, rand(t, n2+1))
end
=#

"""
    Hierarchical_Gumbel_cop

Fields:
- n::Int - number of marginals
- θ::Vector{Float64} - vector of parameters, must be decreasing  and θ[end] ≧ 1, for the
sufficient nesting condition to be fulfilled.

The hierarchically nested Gumbel copula C_θₙ₋₁(C_θₙ₋₂( ... C_θ₂(C_θ₁(u₁, u₂), u₃)...uₙ₋₁) uₙ)

Constructor

    Hierarchical_Gumbel_cop(θ::Vector{Float64})

Constructor

    Hierarchical_Gumbel_cop(ρ::Vector{Float64}, cor::String)
uses cor = "Kendall" or "Spearman" correlation to compute θ

```jldoctest

julia> c = Hierarchical_Gumbel_cop([5., 4., 3.])
Hierarchical_Gumbel_cop(4, [5.0, 4.0, 3.0])

julia> c = Hierarchical_Gumbel_cop([0.95, 0.5, 0.05], "Kendall")
Hierarchical_Gumbel_cop(4, [19.999999999999982, 2.0, 1.0526315789473684])
```
"""
struct Hierarchical_Gumbel_cop
  n::Int
  θ::Vector{Float64}
  function(::Type{Hierarchical_Gumbel_cop})(θ::Vector{Float64})
      testθ(θ[end], "gumbel")
      issorted(θ; rev=true) || throw(DomainError("violated sufficient nesting condition, parameters must be descending"))
      new(length(θ)+1, θ)
  end
  function(::Type{Hierarchical_Gumbel_cop})(ρ::Vector{Float64}, cor::String)
      θ = map(i -> getθ4arch(ρ[i], "gumbel", cor), 1:length(ρ))
      issorted(θ; rev=true) || throw(DomainError("violated sufficient nesting condition, parameters must be descending"))
      new(length(θ)+1, θ)
  end
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
function simulate_copula(t::Int, copula::Hierarchical_Gumbel_cop; rng::AbstractRNG = Random.GLOBAL_RNG)
  θ = copula.θ
  n = copula.n
  θ = vcat(θ, [1.])
  U = zeros(t, n)
  for j in 1:t
      X = rand(rng)
      for i in 1:(n-1)
          X = gumbel_step(vcat(X, rand(rng)), θ[i], θ[i+1] ; rng = rng)
      end
     U[j,:] = X
    end
    U
end

"""
    nested_gumbel_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64},
                         θ::Float64, rand_vec::Vector{Float64}; rng::AbstractRNG)

"""
function nested_gumbel_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64},
                         θ::Float64, rand_vec::Vector{Float64}; rng::AbstractRNG)
    V0 = levyel(θ, rand(rng), rand(rng))
    u = copy(rand_vec)
    for i in 1:length(n)
      u[n[i]] = gumbel_step(rand_vec[n[i]], ϕ[i], θ; rng = rng)
    end
    u = -log.(u)./V0
    return exp.(-u.^(1/θ))
end

"""
    nested_amh_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64},
                         θ::Float64, rand_vec::Vector{Float64}; rng::AbstractRNG)
"""
function nested_amh_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64},
                         θ::Float64, rand_vec::Vector{Float64}; rng::AbstractRNG)
    V0 = 1 .+ quantile.(Geometric(1-θ), rand_vec[end])
    u = copy(rand_vec[1:end-1])
    for i in 1:length(n)
      u[n[i]] = amh_step(rand_vec[n[i]], V0, ϕ[i], θ; rng = rng)
    end
    u = -log.(u)./V0
    return (1-θ) ./(exp.(u) .-θ)
end

"""
    nested_frank_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64}, θ::Float64, rand_vec::Vector{Float64}, logseries::Vector{Float64},
                         logseries_children::Vector{Vector{Float64}};
                         rng::AbstractRNG)
"""
function nested_frank_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64},
                         θ::Float64, rand_vec::Vector{Float64}, logseries::Vector{Float64},
                         logseries_children::Vector{Vector{Float64}};
                         rng::AbstractRNG)
    V0 = findlast(logseries .< rand_vec[end])
    u = copy(rand_vec[1:end-1])
    for i in 1:length(n)
      u[n[i]] = frank_step(rand_vec[n[i]], V0, ϕ[i], θ, logseries_children[i]; rng = rng)
    end
    u = -log.(u)./V0
    return -log.(1 .+exp.(-u) .*(exp(-θ)-1)) ./θ
end

"""
    nested_clayton_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64}, θ::Float64, rand_vec::Vector{Float64}; rng::AbstractRNG = Random.GLOBAL_RNG)
"""
function nested_clayton_gen(n::Vector{Vector{Int}}, ϕ::Vector{Float64},
                         θ::Float64, rand_vec::Vector{Float64}; rng::AbstractRNG)
    V0 = quantile.(Gamma(1/θ, 1), rand_vec[end])
    u = copy(rand_vec[1:end-1])
    for i in 1:length(n)
      u[n[i]] = clayton_step(rand_vec[n[i]], V0, ϕ[i], θ; rng = rng)
    end
    u = -log.(u)./V0
    return (1 .+ u).^(-1/θ)
end

"""
    gumbel_step(u::Vector{Float64}, ϕ::Float64, θ::Float64; rng::AbstractRNG)
"""
function gumbel_step(u::Vector{Float64}, ϕ::Float64, θ::Float64; rng::AbstractRNG)
    u = -log.(u)./levyel(ϕ/θ, rand(rng), rand(rng))
    return exp.(-u.^(θ/ϕ))
end

"""
    clayton_step(u::Vector{Float64}, V0::Float64, ϕ::Float64, θ::Float64; rng::AbstractRNG)
"""
function clayton_step(u::Vector{Float64}, V0::Float64, ϕ::Float64, θ::Float64; rng::AbstractRNG)
    u = -log.(u)./tiltedlevygen(V0, ϕ/θ; rng = rng)
    return exp.(V0.-V0.*(1 .+u).^(θ/ϕ))
end

"""
    frank_step(u::Vector{Float64}, V0::Int, ϕ::Float64, θ::Float64, logseries_child::Vector{Float64}; rng::AbstractRNG)
"""
function frank_step(u::Vector{Float64}, V0::Int, ϕ::Float64, θ::Float64, logseries_child::Vector{Float64}; rng::AbstractRNG)
    u = -log.(u)./nestedfrankgen(ϕ, θ, V0, logseries_child; rng = rng)
    X = (1 .-(1 .-exp.(-u)*(1-exp(-ϕ))).^(θ/ϕ))./(1-exp(-θ))
    return X.^V0
end
"""
    amh_step(u::Vector{Float64}, V0::Float64, ϕ::Float64, θ::Float64; rng::AbstractRNG)
"""
function amh_step(u::Vector{Float64}, V0::Float64, ϕ::Float64, θ::Float64; rng::AbstractRNG)
    w = quantile(NegativeBinomial(V0, (1-ϕ)/(1-θ)), rand(rng))
    u = -log.(u)./(V0 + w)
    X = ((exp.(u) .-ϕ) .*(1-θ) .+θ*(1-ϕ)) ./(1-ϕ)
    return X.^(-V0)
end

"""
nestedcopulag(copula::String, ns::Vector{Vector{Int}}, ϕ::Vector{Float64}, θ::Float64, r::Matrix{Float64})

Given [0,1]ᵗˣˡ ∋ r, returns t realizations of l-1 variate data from nested archimedean copula


```jldoctest
julia> Random.seed!(43)

julia> nestedcopulag("clayton", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
julia> nestedcopulag("clayton", [[1,2],[3,4]], [2., 3.], 1.1, [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6])
2×4 Array{Float64,2}:
 0.153282  0.182421  0.374228  0.407663
 0.69035   0.740927  0.254842  0.279192
```
"""
function nestedcopulag(copula::String, ns::Vector{Vector{Int}}, ϕ::Vector{Float64}, θ::Float64,
                                                        r::Matrix{Float64})
    rng = Random.GLOBAL_RNG
    t = size(r,1)
    n = size(r,2)-1
    u = zeros(t, n)
    if copula == "clayton"
        for j in 1:t
            u[j,:] = nested_clayton_gen(ns, ϕ, θ, r[j,:]; rng = rng)
        end
    elseif copula == "amh"
        for j in 1:t
            u[j,:] = nested_amh_gen(ns, ϕ, θ, r[j,:]; rng = rng)
        end
    elseif copula == "frank"
        ws = [logseriescdf(1-exp(theta)) for theta in ϕ]
        w = logseriescdf(1-exp(-θ))
        for j in 1:t
            u[j,:] = nested_frank_gen(ns, ϕ, θ, r[j,:], w, ws; rng = rng)
        end
    elseif copula == "gumbel"
        v = r[:,end]
        p = invperm(sortperm(v))
        V0 = [levyel(θ, rand(rng), rand(rng)) for i in 1:t]
        V0 = sort(V0)[p]
        for j in 1:t
            rand_vec = r[j,1:end-1]
            x = copy(rand_vec)
            for i in 1:length(ϕ)
              x[ns[i]] = gumbel_step(rand_vec[ns[i]], ϕ[i], θ; rng = rng)
            end
            x = -log.(x)./V0[j]
            u[j,:] = exp.(-x.^(1/θ))
        end
    end
    return u
end

#=
"""
  nestedstep(copula::String, u::Matrix{Float64}, V0::Union{Vector{Float64}, Vector{Int}}, ϕ::Float64, θ::Float64)

Given u ∈ [0,1]ᵗⁿ and V0 ∈ ℜᵗ returns u ∈ [0,1]ᵗⁿ for a given archimedean nested copula with
inner copulas parameters ϕ anu auter copula parameter θ

```jldoctest
julia> nestedstep("clayton", [0.2 0.8; 0.1 0.7], [0.2, 0.4], 2., 1.5)
2×2 Array{Float64,2}:
 0.283555  0.789899
 0.322614  0.806915
```
"""
function nestedstep(copula::String, u::Matrix{Float64}, V0::Union{Vector{Float64}, Vector{Int}},
                                                        ϕ::Float64, θ::Float64)
  if copula == "amh"
    w = [quantile(NegativeBinomial(v, (1-ϕ)/(1-θ)), rand()) for v in V0]
    u = -log.(u)./(V0 + w)
    X = ((exp.(u) .-ϕ) .*(1-θ) .+θ*(1-ϕ)) ./(1-ϕ)
    return X.^(-V0)
  elseif copula == "frank"
    u = -log.(u)./nestedfrankgen(ϕ, θ, V0)
    X = (1 .-(1 .-exp.(-u)*(1-exp(-ϕ))).^(θ/ϕ))./(1-exp(-θ))
    return X.^V0
  elseif copula == "clayton"
    u = -log.(u)./tiltedlevygen(V0, ϕ/θ)
    return exp.(V0.-V0.*(1 .+u).^(θ/ϕ))
  elseif copula == "gumbel"
    u = -log.(u)./levygen(ϕ/θ, rand(length(V0)))
    return exp.(-u.^(θ/ϕ))
  end
  throw(AssertionError("$(copula) not supported"))
end
=#


#=
"""
  testnestedθϕ(n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String)

Tests parameters, its hierarchy and size of parametes vector for nested archimedean copulas.
"""
function testnestedθϕ(n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String)
  testθ(θ, copula)
  map(p -> testθ(p, copula), ϕ)
  θ <= minimum(ϕ) || throw(DomainError("wrong heirarchy of parameters"))
  length(n) == length(ϕ) || throw(AssertionError("number of subcopulas ≠ number of parameters"))
  (copula != "clayton") | (maximum(ϕ) < θ+2*θ^2+750*θ^5) || warn("θ << ϕ for clayton nested copula, marginals may not be uniform")
end
=#
