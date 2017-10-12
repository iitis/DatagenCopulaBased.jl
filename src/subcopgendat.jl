"""
  rand2cop(u1::Vector{Float64}, θ::Union{Int, Float64}, copula::String)

Returns vector of data generated using copula::String given vector of uniformly
distributed u1 and copula parameter θ.
"""

function rand2cop(u1::Vector{Float64}, θ::Union{Int, Float64}, copula::String)
  w = rand(length(u1))
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
  frankcopulagen(t::Int, θ::Vector{Float64})

Returns: t x n Matrix{Float}, t realisations of n variate data, where n = length(θ)+1.
To generate data uses Frank bivariate sub-copulas with parameters θᵢ ≠ 0 for each
neighbour marginals (i'th and i+1'th). If pearsonrho = true, parameters
are Pearson correlation coefficents fulfilling (-1 > θᵢ > 1) ∧ (θᵢ ≠ 0).

```jldoctest
julia> srand(43);

julia> frankcopulagen(10, [4., 11.])
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
function frankcopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false)
  u = rand(t, 1)
  !(0. in θ) || throw(AssertionError("not supported for θ parameter = 0"))
  if pearsonrho
    maximum(abs.(θ)) < 1 || throw(AssertionError("correlation must be in range (-1, 1)"))
    θ = map(r -> ρ2θ(r, "frank"), θ)
  end
  for i in 1:length(θ)
    u = hcat(u, rand2cop(u[:, i], θ[i], "frank"))
  end
  u
end

"""

  claytoncopulagen(t::Int = 1000, θ::Vector{Float64}; pearsonrho, reverse)

Returns: t x n Matrix{Float}, t realisations of n variate data, where n = length(θ)+1.
To generate data uses Clayton bivariate sub-copulas with parameters (θᵢ ≥ -1) ^ ∧ (θᵢ ≠ 0).
If pearsonrho = true parameters are Pearson correlation coefficents
fulfulling (-1 > θᵢ > 1) ∧ (θᵢ ≠ 0)
If reversed = true, returns data from reversed Clayton bivariate subcopulas.

```jldoctest
julia> srand(43);

julia> claytoncopulagen(9, [-0.9, 0.9, .7]; pearsonrho = true)
9×4 Array{Float64,2}:
 0.180975  0.942164   0.872673   0.970384
 0.775377  0.230724   0.340819   0.347218
 0.888934  0.0579034  0.190519   0.441244
 0.924876  0.0360802  0.0294198  0.0368123
 0.408278  0.461712   0.889275   0.850588
 0.912603  0.0433313  0.0315759  0.0300926
 0.828727  0.270476   0.274191   0.381278
 0.400537  0.469634   0.633396   0.405873
 0.429437  0.440285   0.478058   0.640165
```
"""

function claytoncopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
  minimum(θ) >= -1 || throw(AssertionError("not supported for parameter < -1"))
  !(0. in θ) || throw(AssertionError("not supported for θ parameter = 0"))
  if pearsonrho
    maximum(θ) < 1 || throw(AssertionError("correlation coeficient must be in range (-1,1)"))
    θ = map(r -> ρ2θ(r, "clayton"), θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    u = hcat(u, rand2cop(u[:, i], θ[i], "clayton"))
  end
  reverse? 1-u : u
end

"""

  amhcopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool, reverse::Bool)

Returns: t x n Matrix{Float}, t realisations of n variate data, where n = length(θ)+1.
To generate data uses Ali-Mikhail-Haq bivariate sub-copulas with parameters -1 ≥ θᵢ ≥ 1.
If pearsonrho = true parameters are Pearson correlation coefficents fulfilling -0.2816 > θᵢ >= .5.
If reversed = true returns data from reversed Ali-Mikhail-Haq bivariate sub-copulas.

```jldoctest
julia> srand(43);

julia> amhcopulagen(10, [1, 0.3])
10×3 Array{Float64,2}:
 0.180975  0.441152   0.996646
 0.775377  0.225086   0.177177
 0.888934  0.327726   0.977642
 0.924876  0.291837   0.108233
 0.408278  0.187564   0.402945
 0.912603  0.848985   0.830843
 0.828727  0.0571042  0.471947
 0.400537  0.0758159  0.913208
 0.429437  0.527526   0.405125
 0.955881  0.919363   0.838458
```
"""

function amhcopulagen(t::Int, θ::Vector{Float64}; pearsonrho::Bool = false, reverse::Bool = false)
  minimum(θ) >= -1 || throw(AssertionError("not supported for parameter < -1"))
  maximum(θ) <= 1 || throw(AssertionError("not supported for parameter > 1"))
  if pearsonrho
    maximum(θ) <= 0.5 || throw(AssertionError("not supported for correlation > 0.5"))
    minimum(θ) > -0.2816 || throw(AssertionError("not supported for correlation <= -0.2816"))
    θ = map(r -> ρ2θ(r, "amh"), θ)
  end
  u = rand(t,1)
  for i in 1:length(θ)
    u = hcat(u, rand2cop(u[:, i], θ[i], "amh"))
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
  subcopdatagen(t::Int, n::Int, cli::Array, sti::Array, std::Vector)

Returns Matrix{Float} t x n of t realisations of n variate random variable with gaussian marginals
with (0, std[i]) parameters. Data have generally gaussian copula, clayton subcopula at
cli indices and tstudent copula at sti indices. Obviously 0 .< cli .<= n and  0 .< sli .<= n
"""

VVI = Vector{Vector{Int}}

function copulamixbv(t::Int, n::Int = 30, cli::VVI = [[]], fi::VVI = [[]], amhi::VVI = [[]], ti::Array = [])
  Σ = cormatgen(n, 0.8,true,true)
  z = gausscopulagen(t, Σ)
  if cli !=[]
    for i in 1:length(cli)
      j = cli[i]
      θ = ρ2θ(Σ[j[1],j[2]], "clayton")
      z[:,j[2]] = rand2cop(z[:,j[1]], θ, "clayton")
    end
    for i in 1:length(fi)
      j = fi[i]
      θ = ρ2θ(Σ[j[1],j[2]], "frank")
      z[:,j[2]] = rand2cop(z[:,j[1]], θ, "frank")
    end
    for i in 1:length(amhi)
      j = amhi[i]
      ρ = Σ[j[1],j[2]]
      θ = 1.
      #println(ρ)
      if ρ < -0.28
        θ = -1.
      elseif ρ < 0.5
        θ = ρ2θ(ρ, "amh")
      end
      #println(θ)
      z[:,j[2]] = rand2cop(z[:,j[1]], θ, "amh")
    end
  end
  (ti == [])? (): g2tsubcopula!(z, Σ, tcinds)
  z, Σ
end


function copulamix(t::Int, n::Int = 30, nunumfc::Bool = true, cli::Array = [],
                                                              amhi::Array = [],
                                                              gi::Array = [],
                                                              fri::Array = [],
                                                              mo::Array = [],
                                                              λ::Array = [0.8, 0.1],
                                                              ti::Array = [])
  Σ = cormatgen(n, 0.5, nunumfc, false)
  x = transpose(rand(MvNormal(Σ),t))
  xgauss = copy(x)
  x = cdf(Normal(0,1), x)
  j = 1
  cop = ["clayton", "amh", "gumbel", "frank", "Marshal-Olkin"]
  for ind in [cli, amhi, gi, fri, mo]
    if ind != []
      l = length(ind)
      i = ind
      #lim =(cop[j] =="gumbel")? l+2: l+1
      lim = l+1
      if cop[j] =="Marshal-Olkin"
        lim = 2^(length(ind))-1
      end
      for p in 0:(lim-l-1)
        k = p%l+1
        i = vcat(i, find(Σ[:, k].== maximum(Σ[setdiff(collect(1:n),i),k])))
      end
      a, s = eig(Σ[i,i])
      w = xgauss[:, i]*s./transpose(sqrt.(a))
      w[:, end] = sign(cov(xgauss[:, ind[1]], w[:, end]))*w[:, end]
      v = cdf(Normal(0,1), w)
      if cop[j] == "Marshal-Olkin"
        ρ = [Σ[ind[k[1]], ind[k[2]]] for k in collect(combinations(1:l,2))]
        x[:,ind] = mocopula(v, l, τ2λ(ρ, λ))
      else
        θ = ρ2θ(Σ[ind[1], ind[2]], cop[j])
        x[:,ind] = copulagen(cop[j], v, θ)
      end
    end
    j += 1
  end
  (ti == [])? (): g2tsubcopula!(x, Σ, ti)
  x, Σ
end


function copulamixgenunif(t::Int, n::Int = 30, nunumfc::Bool = true, cli::Array = [], fri::Array = [], amhi::Array = [], ti::Array = [])
  Σ = cormatgen(n, 0.8, nunumfc, false)
  x = transpose(rand(MvNormal(Σ),t))
  v = -x*eig(Σ)[2][:,end]
  z = cdf(Normal(0,1), x)
  vx = cdf(Normal(0,std(v)), v)
  if cli !=[]
    for i in 1:length(cli)
      θ = ρ2θ(cor(v, x[:,cli[i]]), "clayton")
      z[:,cli[i]] = rand2cop(vx, θ, "clayton")
    end
    for i in 1:length(fri)
      θ = ρ2θ(cor(v, x[:,fri[i]]), "frank")
      z[:,fri[i]] = rand2cop(vx, θ, "frank")
    end
    for i in 1:length(amhi)
      ρ = cor(v, x[:,amhi[i]])
      θ = 1.
      println(ρ)
      if ρ < -0.28
        θ = -1.
      elseif ρ < 0.5
        θ = ρ2θ(ρ, "amh")
      end
      println(θ)
      z[:,amhi[i]] = rand2cop(vx, θ, "amh")
    end
  end
  (ti == [])? (): g2tsubcopula!(z, Σ, ti)
  z, Σ
end
