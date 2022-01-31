# transforms marginal univariate distributions

VecVec{T} = Union{Vector{Vector{Int64}}, Vector{Vector{T}}}

"""
  convertmarg!(X::Matrix, d::UnionAll, p::Vector{Vector})

Takes matrix X of realizations of size(X,2) = n dimensional random variable, with
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
 -0.177808Real4172
  1.70477    10.4192
```
"""
function convertmarg!(U, d, p = [fill([0,1], size(U, 2))...];
                                                testunif = true)
  for i = 1:size(U, 2)
    if testunif
      pvalue(ExactOneSampleKSTest(U[:,i],Uniform(0,1)))>0.0001 || throw(AssertionError("$i marg. not unif."))
    end
    @inbounds U[:,i] = quantile.(d(p[i]...), U[:,i])
  end
end

  # generates covariance matrix

"""
    cormatgen(n::Int = 20)

Returns symmetric correlation matrix Σ of size n x n.
Method:

    a = rand(n,n)
    b = a*a'
    c = b./maximum(b)

Example:

```jldoctest
julia> Random.seed!(43);

julia> cormatgen(2)
2×2 Array{Float64,2}:
 1.0       0.660768
 0.660768  1.0
```
"""
function cormatgen(n = 20)
  a = rand(n,n)
  b = a*a'
  c = b./maximum(b)
  c .- Matrix(Diagonal(c)) .+ Matrix(1.0I, size(c)...)

end

"""
    cormatgen_rand(n::Int = 20)

Returns symmetric correlation matrix Σ of size n x n.
Method:

    a = rand(n,n)
    b = a*a'
    diagb = Matrix(Diagonal(1 ./sqrt.(LinearAlgebra.diag(b))))
    b = diagb*b*diagb

In general gives higher correlations than the cormatgen(). Example:

```jldoctest
julia> Random.seed!(43);

julia> cormatgen_rand(2)
2×2 Array{Float64,2}:
 1.0       0.879086
 0.879086  1.0
```
"""
function cormatgen_rand(n = 20)
  a = rand(n,n)
  b = a*a'
  diagb = Matrix(Diagonal(1 ./sqrt.(LinearAlgebra.diag(b))))
  b = diagb*b*diagb
  (b+b')/2
end

"""
    cormatgen_constant(n::Int, α::Real)

Returns the constant correlation matrix with constant correlations equal to 0 <= α <= 1

```julia
julia> cormatgen_constant(2, 0.5)
2×2 Array{Real,2}:
 1.0  0.5
 0.5  1.0
```
"""
function cormatgen_constant(n, α)
  @assert 0 <= α <= 1 "α should satisfy 0 <= α <= 1"
  α .*ones(n, n) .+(1-α) .*Matrix(1.0I, n,n)
end

function random_unit_normal_vector(dim)
  result = rand(Normal(), dim, 1)
  result /= norm(result)
end

"""
    cormatgen_constant_noised(n::Int, α::Real; ϵ::Real = (1 .-α)/2.)

Returns the constant correlation matrix of size n x n  with constant correlations equal to 0 <= α <= 1
and additinal noise determinde by ϵ.

```julia
julia> Random.seed!(43);

julia> cormatgen_constant_noised(3, 0.5)
3×3 Array{Float64,2}:
 1.0       0.506271  0.285793
 0.506271  1.0       0.475609
 0.285793  0.475609  1.0
```
"""
function cormatgen_constant_noised(n, α; ϵ = (1 .-α)/2.)
  @assert 0 <= ϵ <= 1-α "ϵ must satisfy 0 <= ϵ <= 1-α"
  result = cormatgen_constant(n, α)
  u = hcat([random_unit_normal_vector(n) for i=1:n]...)
  result += ϵ .*(u'*u)
  result - ϵ .*Matrix(1.0I, size(result)...)
end

"""
    cormatgen_two_constant(n::Int, α::Real, β::Real)

Returns the correlation matrix of size n x n of correlations determined by two constants, first must be greater than the second.

```julia
julia> cormatgen_two_constant(6, 0.5, 0.1)
6×6 Array{Float64,2}:
 1.0  0.5  0.5  0.1  0.1  0.1
 0.5  1.0  0.5  0.1  0.1  0.1
 0.5  0.5  1.0  0.1  0.1  0.1
 0.1  0.1  0.1  1.0  0.1  0.1
 0.1  0.1  0.1  0.1  1.0  0.1
 0.1  0.1  0.1  0.1  0.1  1.0


 julia> cormatgen_two_constant(4, 0.5, 0.1)
 4×4 Array{Float64,2}:
  1.0  0.5  0.1  0.1
  0.5  1.0  0.1  0.1
  0.1  0.1  1.0  0.1
  0.1  0.1  0.1  1.0
```
"""
function cormatgen_two_constant(n, α, β)
  @assert α > β "First argument must be greater"
  result = fill(β, (n,n))
  result[1:div(n,2),1:div(n,2)] = fill(α, (div(n,2),div(n,2)))
  result += Matrix(1.0I, size(result)...) - Matrix(Diagonal(result))
  result
end

"""
    cormatgen_two_constant_noised(n::Int, α::Real, β::Real; ϵ::Real= (1-α)/2)

Returns the correlation matrix of size n x n  of correlations determined by two constants, first must be greater than the second.
Additional noise is introduced by the parameter ϵ.

```julia
julia> Random.seed!(43);

julia> cormatgen_two_constant_noised(4, 0.5, 0.1)
4×4 Array{Float64,2}:
  1.0         0.314724   0.190368  -0.0530078
  0.314724    1.0       -0.085744   0.112183
  0.190368   -0.085744   1.0        0.138089
 -0.0530078   0.112183   0.138089   1.0
```
"""
function cormatgen_two_constant_noised(n, α, β; ϵ= (1-α)/2)
  @assert ϵ <= 1-α
  result = cormatgen_two_constant(n, α, β)
  u = hcat([random_unit_normal_vector(n) for i=1:n]...)
  result += ϵ .*(u'*u)
  result - ϵ .*Matrix(1.0I, size(result)...)
end

"""
    cormatgen_toeplitz(n::Int, ρ::Real)

Returns the correlation matrix of size n x n of the Toeplitz structure with
maximal correlation equal to ρ.

```julia
julia> cormatgen_toeplitz(5, 0.9)
5×5 Array{Float64,2}:
 1.0     0.9    0.81  0.729  0.6561
 0.9     1.0    0.9   0.81   0.729
 0.81    0.9    1.0   0.9    0.81
 0.729   0.81   0.9   1.0    0.9
 0.6561  0.729  0.81  0.9    1.0

julia> cormatgen_toeplitz(5, 0.6)
5×5 Array{Float64,2}:
 1.0     0.6    0.36  0.216  0.1296
 0.6     1.0    0.6   0.36   0.216
 0.36    0.6    1.0   0.6    0.36
 0.216   0.36   0.6   1.0    0.6
 0.1296  0.216  0.36  0.6    1.0
```
"""
function cormatgen_toeplitz(n, ρ)
  @assert 0 <= ρ <= 1 "ρ needs to satisfy 0 <= ρ <= 1"
  [ρ^(abs(i-j)) for i=0:n-1, j=0:n-1]
end

"""
    cormatgen_toeplitz_noised(n::Int, ρ::Real; ϵ=(1-ρ)/(1+ρ)/2)

Returns the correlation matrix of size n x n of the Toeplitz structure with
maximal correlation equal to ρ. Additiona noise id added by the ϵ parameter.

```julia
julia> Random.seed!(43);

julia> cormatgen_toeplitz_noised(5, 0.9)
5×5 Array{Float64,2}:
 1.0       0.89656   0.812152  0.720547  0.64318
 0.89656   1.0       0.918136  0.832571  0.734564
 0.812152  0.918136  1.0       0.915888  0.822804
 0.720547  0.832571  0.915888  1.0       0.903819
 0.64318   0.734564  0.822804  0.903819  1.0
```
"""
function cormatgen_toeplitz_noised(n, ρ; ϵ=(1-ρ)/(1+ρ)/2)
  @assert 0 <= ϵ <= (1-ρ)/(1+ρ) "ϵ must satisfy 0 <= ϵ <= (1-ρ)/(1+ρ)"

  result = cormatgen_toeplitz(n, ρ)
  u = hcat([random_unit_normal_vector(n) for i=1:n]...)
  result += ϵ .*(u'*u)
  result - ϵ .*Matrix(1.0I, size(result)...)
end
