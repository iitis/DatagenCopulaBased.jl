### Following R. Nelsen 'An Introduction to Copulas', Springer Science & Business Media, 1999 - 216,
### for bivariate Archimedean copulas `C(u₁,u₂)` data can be generated as follow:
### * draw `u₁ = rand()`,
### * define `w = ∂C(u₁, u₂)\∂u₁` and inverse `u₂ = f(w, u₁)`,
### * draw  `w = rand()`
### * return a pair u₁, u₂.

### This method can be applied in practice for Clayton, Frank and Ali-Mikhail-Haq copula. If we use
### this method recursively, we can get `n`-variate data with uniform marginals on
### `[0,1]`, where each neighbour pair
### of marginals `uᵢ uⱼ` for `j = i+1` are draw form a bivariate subcopula with
### parameter `θᵢ`, the only condition for `θᵢ`
### is such as for a corresponding bivariate copula.



"""
  rand2cop(u1::Vector{Float64}, θ::Union{Int, Float64}, copula::String)

Returns vector of data generated using copula::String given vector of uniformly
distributed u1 and copula parameter θ.
"""

function rand2cop(u1::Vector{Float64}, θ::Union{Int, Float64}, copula::String)
  w = rand(length(u1))
  copula in ["clayton", "amh", "frank"] || throw(AssertionError("$(copula) copula is not supported"))
  if copula == "clayton"
    return (u1.^(-θ).*(w.^(-θ/(1+θ))-1)+1).^(-1/θ)
  elseif copula == "frank"
    return -1/θ*log.(1+(w*(1-exp(-θ)))./(w.*(exp.(-θ*u1)-1)-exp.(-θ*u1)))
  elseif copula == "amh"
    a = 1-u1
    b = 1-θ.*(1+2*a.*w)+2*θ^2*a.^2.*w
    c = 1-θ.*(2-4*w+4*a.*w)+θ.^2.*(1-4*a.*w+4*a.^2.*w)
    return 2*w.*(a*θ-1).^2./(b+sqrt.(c))
  end
end

"""
  chaincopulagen(t::Int, θ::Union{Vector{Float64}, Vector{Int}}, copula::String;
                                              rev::Bool = false, cor::String = "")

Returns: t x n Matrix{Float}, t realisations of n variate data, where n = length(θ)+1.
To generate data uses chain of Archimedean one parameter bivariate copula for each
neighbour marginals (i'th and i+1'th).

Following copula families are supported: clayton, frank and amh -- Ali-Mikhail-Haq.

If rev == true, reverse the copula output i.e. u → 1-u (we call it reversed copula).
It cor == pearson, kendall, uses correlation coeficient as a parameter

```jldoctest
julia> srand(43);

julia> chaincopulagen(10, [4., 11.], "frank")
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
"""


VFI = Union{Vector{Float64}, Vector{Int}}

function chaincopulagen(t::Int, θ::VFI, copula::String; rev::Bool = false, cor::String = "")
  if ((cor == "pearson") | (cor == "kendall"))
    θ = map(i -> usebivρ(i, copula, cor), θ)
  else
    map(i -> testbivθ(i, copula), θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    u = hcat(u, rand2cop(u[:, i], θ[i], copula))
  end
  rev? 1-u : u
end

"""
  testbivθ(θ::Union{Float64}, copula::String)

Tests bivariate copula parameter

clayton bivariate sub-copulas with parameters (θᵢ ≥ -1) ^ ∧ (θᵢ ≠ 0).
amh -- Ali-Mikhail-Haq bivariate sub-copulas with parameters -1 ≥ θᵢ ≥ 1
frank bivariate sub-copulas with parameters (θᵢ ≠ 0)
"""
function testbivθ(θ::Union{Float64, Int}, copula::String)
  !(0. in θ)|(copula == "amh") || throw(AssertionError("not supported for θ = 0"))
  if copula == "clayton"
    θ >= -1 || throw(AssertionError("not supported for θ < -1"))
  elseif copula == "amh"
    -1 <= θ <= 1|| throw(AssertionError("amh biv. copula supported only for -1 ≤ θ ≤ 1"))
  end
end

"""
Returns Float64, a copula parameter given a pearson or kendall correlation

For clayton or frank copula correlation fulfulling (-1 > ρᵢ > 1) ∧ (ρᵢ ≠ 0)
For amh copula pearson correlation fulfilling -0.2816 > ρᵢ >= .5. while kendall -0.18 < τ < 1/3
"""

function usebivρ(ρ::Float64, copula::String, cor::String)
  if copula == "amh"
      -0.2816 < ρ <= 0.5 || throw(AssertionError("correlation coeficiant must fulfill -0.2816 < ρ <= 0.5"))
    if cor == "kendall"
      -0.18 < ρ < 1/3 || throw(AssertionError("correlation coeficiant must fulfill -0.2816 < ρ <= 1/3"))
    end
  else
    -1 < ρ < 1 || throw(AssertionError("correlation coeficiant must fulfill -1 < ρ < 1"))
    !(0. in ρ) || throw(AssertionError("not supported for ρ = 0"))
  end
  (cor == "kendall")? τ2θ(ρ, copula): ρ2θ(ρ, copula)
end


# chain frechet copulas

"""
  chainfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))

Retenares data from nested hierarchical frechet copula with parameters
vectors α and β, such that ∀ᵢ 0 α[i] + β[i] ≤1 α[i] > 0, and β[i] > 0 |α| = |β|

```jldoctest
julia> srand(43)

julia> julia> chainfrechetcopulagen(10, [0.6, 0.4], [0.3, 0.5])
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

function chainfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zeros(α))
  length(α) == length(β) || throw(AssertionError("different lengths of parameters"))
  minimum(α) >= 0 || throw(AssertionError("negative α parameter"))
  minimum(β) >= 0 || throw(AssertionError("negative β parameter"))
  maximum(α+β) <= 1 || throw(AssertionError("α[i] + β[i] > 0"))
  fncopulagen(α, β, rand(t, length(α)+1))
end

"""

  fncopulagen(α::Vector{Float64}, β::Vector{Float64}, u::Matrix{Float64})


```jldoctest

julia> fncopulagen(2, [0.2, 0.4], [0.1, 0.1], [0.2 0.4 0.6; 0.3 0.5 0.7])
2×3 Array{Float64,2}:
 0.6  0.4  0.2
 0.7  0.5  0.3

```
"""

function fncopulagen(α::Vector{Float64}, β::Vector{Float64}, u::Matrix{Float64})
  p = invperm(sortperm(u[:,1]))
  u = u[:,end:-1:1]
  lx = floor.(Int, size(u,1).*α)
  li = floor.(Int, size(u,1).*β) + lx
  for j in 1:size(u, 2)-1
    u[p[1:lx[j]],j+1] = u[p[1:lx[j]], j]
    r = p[lx[j]+1:li[j]]
    u[r,j+1] = 1-u[r,j]
  end
  u
end
