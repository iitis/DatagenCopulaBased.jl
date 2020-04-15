
# transforms marginal univariate distributions
"""
  convertmarg!(X::Matrix, d::UnionAll, p::Vector{Vector})

Takes matrix X of realisations of size(X,2) = n dimensional random variable, with
uniform marginals numbered by i, and convert those marginals to common distribution
d with parameters p[i].
If `testunif = true` each marginal is tested for uniformity.

```jldoctest
julia> Random.seed!(43);

julia> x = rand(10,2);

julia> convertmarg!(x, Normal, [[0, 1],[0, 10]])

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
"""
VecVec = Union{Vector{Vector{Int64}}, Vector{Vector{Float64}}}

function convertmarg!(U::Matrix{T}, d::UnionAll, p::VecVec = [fill([0,1], size(U, 2))...];
                                                testunif::Bool = true) where T <: AbstractFloat
  for i = 1:size(U, 2)
    if testunif
      pvalue(ExactOneSampleKSTest(U[:,i],Uniform(0,1)))>0.0001 || throw(AssertionError("$i marg. not unif."))
    end
    @inbounds U[:,i] = quantile.(d(p[i]...), U[:,i])
  end
end

  # generates covariance matrix

"""
    cormatgen(n::Int, ρ::Float64 = 0.5, ordered::Bool = false, altersing::Bool = true)

Returns symmetric correlation matrix Σ of size n x n, with reference correlation 0 < ρ < 1.
If ordered = false, Σ elements varies arround ρ, i.e. σᵢⱼ ≈ ρ+δ else they drop
as indices differences rise, i.e. σᵢⱼ ≳ σᵢₖ as |i-j| < |i-k|.
If altersing = true, some σ are positive and some negative, else ∀ᵢⱼ σᵢⱼ ≥ 0.

```jldoctest
julia> Random.seed!(43);

julia> julia> cormatgen(2)
2×2 Array{Float64,2}:
 1.0       0.660768
 0.660768  1.0
```
"""
function cormatgen(n::Int = 20)
  a = rand(n,n)
  b = a*a'
  c = b./maximum(b)
  c .- Matrix(Diagonal(c)) .+ Matrix(1.0I, size(c)...)

end

function cormatgen_rand(n::Int = 20)
  a = rand(n,n)
  b = a*a'
  diagb = Matrix(Diagonal(1 ./sqrt.(LinearAlgebra.diag(b))))
  b = diagb*b*diagb
  (b+b')/2
end

function cormatgen_constant(n::Int, α::Float64)
  @assert 0 <= α <= 1 "α should satisfy 0 <= α <= 1"
  α .*ones(n, n) .+(1-α) .*Matrix(1.0I, n,n)
end

function random_unit_vector(dim::Int)
  result = rand(Normal(), dim, 1)
  result /= norm(result)
end

function cormatgen_constant_noised(n::Int, α::Float64; ϵ::Float64 = (1 .-α)/2.)
  @assert 0 <= ϵ <= 1-α "ϵ must satisfy 0 <= ϵ <= 1-α"
  result = cormatgen_constant(n, α)
  u = hcat([random_unit_vector(n) for i=1:n]...)
  result += ϵ .*(u'*u)
  result - ϵ .*Matrix(1.0I, size(result)...)
end

function cormatgen_two_constant(n::Int, α::Float64, β::Float64)
  @assert α > β "First argument must be greater"
  result = fill(β, (n,n))
  result[1:div(n,2),1:div(n,2)] = fill(α, (div(n,2),div(n,2)))
  result += Matrix(1.0I, size(result)...) - Matrix(Diagonal(result))
  result
end

function cormatgen_two_constant_noised(n::Int, α::Float64, β::Float64; ϵ::Float64= (1-α)/2)
  @assert ϵ <= 1-α
  result = cormatgen_two_constant(n, α, β)
  u = hcat([random_unit_vector(n) for i=1:n]...)
  result += ϵ .*(u'*u)
  result - ϵ .*Matrix(1.0I, size(result)...)
end

function cormatgen_toeplitz(n::Int, ρ::Float64)
  @assert 0 <= ρ <= 1 "ρ needs to satisfy 0 <= ρ <= 1"
  [ρ^(abs(i-j)) for i=0:n-1, j=0:n-1]
end

function cormatgen_toeplitz_noised(n::Int, ρ::Float64; ϵ=(1-ρ)/(1+ρ)/2)
  @assert 0 <= ϵ <= (1-ρ)/(1+ρ) "ϵ must satisfy 0 <= ϵ <= (1-ρ)/(1+ρ)"

  result = cormatgen_toeplitz(n, ρ)
  u = hcat([random_unit_vector(n) for i=1:n]...)
  result += ϵ .*(u'*u)
  result - ϵ .*Matrix(1.0I, size(result)...)
end
