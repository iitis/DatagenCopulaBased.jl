"""

  claytoncopulagen(t::Int = 1000, θ::Vector{Float64}; pearsonrho, reverse)

Returns: t x n Matrix{Float}, t realisations of n-variate data, where n = length(θ)+1.
Each two neighbour marginals (i'th and i+1'th) are generated from bivariate Clayton copula
with parameter θ_i >= -1 ^ θ_i != 0. If pearsonrho parameters -1 > θ_i >= 1 ^ θ_i != 0 are taken as Pearson
correlation coefficents. If reversed returns data from reversed Clayton copula.

```jldoctest
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
```
"""

function claytoncopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
  minimum(θ) >= -1 || throw(AssertionError("not supported for $i th parameter < -1"))
  !(0. in θ) || throw(AssertionError("not supported for $i th parameter = 0"))
  if pearsonrho
    maximum(θ) <= 1 || throw(AssertionError("$i correlation coeficient > 1"))
    θ = map(claytonθ, θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    w = rand(t)
    z = u[:, i]
    u = hcat(u, z.*(w.^(-θ[i]/(1 + θ[i])) - 1 + z.^θ[i]).^(-1/θ[i]))
  end
  reverse? 1-u : u
end


"""
  g2tsubcopula!(z::Matrix{Float}, cormat::Matrix{Float}, subn::Array{Int})

Changes data generated using gaussian copula to data generated using student
 subcopula at indices subn.
"""

function g2tsubcopula!(z::Matrix{Float64}, cormat::Matrix{Float64}, subn::Array{Int}, nu::Int = 10)
  d = Chisq(nu)
  U = rand(d, size(z, 1))
  p = TDist(nu)
  for i in subn
    w = quantile(Normal(0, cormat[i,i]), z[:,i])
    z[:,i] = cdf(p, w.*sqrt.(nu./U))
  end
end

"""
  g2clsubcopula(U::Vector{Float}, ρ::Float)

Returns vector of data generated using clayton (assymatric) copula accoriding to
vector of data U at given pearson correlation coeficient ρ.
"""
function g2clsubcopula(U::Vector{Float64}, ρ::Float64)
  θ = claytonθ(ρ)
  W = rand(length(U))
  U.*(W.^(-θ/(1 + θ)) - 1 + U.^θ).^(-1/θ)
end



"""
  subcopdatagen(t::Int, n::Int, cli::Array, sti::Array, std::Vector)

Returns Matrix{Float} t x n of t realisations of n variate random variable with gaussian marginals
with (0, std[i]) parameters. Data have generally gaussian copula, clayton subcopula at
cli indices and tstudent copula at sti indices. Obviously 0 .< cli .<= n and  0 .< sli .<= n
"""
function subcopdatagen(t::Int, n::Int = 30, cli::Array = [], sti::Array = [], std::Vector{Float64} = [fill(1., n)...])
  cormat = cormatgen(n)
  z = gausscopulagen(t, cormat)
  if cli !=[]
    for i in 2:length(cli)
      z[:,cli[i]] = g2clsubcopula(z[:,cli[i-1]], cormat[cli[i], cli[i-1]])
    end
  end
  if sti !=[]
      g2tsubcopula!(z, cormat, sti)
  end
  convertmarg!(z, Normal, [[0, std[i]] for i in 1:n])
  z
end
