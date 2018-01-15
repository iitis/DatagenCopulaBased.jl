using Cubature
using Distributions
using PyCall
@pyimport matplotlib as mpl
mpl.rc("text", usetex=true)
mpl.use("Agg")
@pyimport matplotlib.ticker as mti
using PyPlot

Cg(x::Vector{Float64}, θ::Union{Int, Float64}) = exp(-((-log(x[1]))^θ+(-log(x[2]))^θ)^(1/θ))

function Ccl(x::Vector{Float64}, θ::Union{Int, Float64})
  if θ > 0
    return (x[1]^(-θ)+x[2]^(-θ)-1)^(-1/θ)
  else
    return (maximum([x[1]^(-θ)+x[2]^(-θ)-1, 0]))^(-1/θ)
  end
end

Cf(x::Vector{Float64}, θ::Union{Int, Float64}) =
  -log(1+((exp(-θ*x[1])-1)*(exp(-θ*x[2])-1))/(exp(-θ)-1))/θ


Camh(x::Vector{Float64}, θ::Union{Int, Float64}) = x[1]*x[2]/(1-θ*(1-x[1])*(1-x[2]))

function getcov(copula::Function, d, θa::Vector{Float64})
  covs = Float64[]
  a = 0.000000001
  b = 0.999999999
  f(y) = pdf.(d, quantile.(d, y))
  for θ in θa
    intf(x) = (copula(x, θ)-x[1]*x[2])/(f(x[1])*f(x[2]))
    push!(covs, hcubature(intf, [a,a],[b,b])[1]/var(d))
  end
  covs
end

function getρ(copula::Function, θa::Vector{Float64})
  covs = Float64[]
  for θ in θa
    push!(covs, 12*hcubature(x-> copula(x, θ), [0,0],[1,1])[1]-3)
  end
  covs
end

θ = [i for i in -0.5:0.1:1.]
function plotcor(copula::Function, θ::Vector{Float64})
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  ax[:plot](θ, getcov(copula, Normal(0,1), θ), label= "normal dist", color = "r", linewidth = 0.5)
  ax[:plot](θ, getcov(copula, LogNormal(1,1), θ), label= "lognormal(1,1)", color = "blue", linewidth = 0.5)
  ax[:plot](θ, getcov(copula, Arcsine(0,1), θ), label= "arcsine(0,1)", color = "green", linewidth = 0.5)
  ax[:plot](θ, getcov(copula, TDist(5), θ), label= "tdist(5)", color = "grey", linewidth = 0.5)
  ax[:plot](θ, getcov(copula, Weibull(1,1), θ), label= "weibull(1,1)", color = "yellow", linewidth = 0.5)
  ax[:plot](θ, getρ(copula, θ), label = "uniform dist", color = "black", linewidth = 0.5)
  ax[:legend](fontsize = 4.5, loc = 4, ncol = 1)
  subplots_adjust(bottom = 0.16, top=0.92, left = 0.18, right = 0.92)
  PyPlot.ylabel("Pearson correlation", labelpad = -1.5)
  PyPlot.xlabel("\$ \\theta \$", labelpad = 0.6)
  fig[:savefig]("pics/cor"*string(copula)*".eps")
end

plotcor(Cg, vcat([1.01], [i for i in 1.2:0.2:10.]))
plotcor(Cf, [i for i in -10.:0.2:10.])
plotcor(Ccl, vcat([i for i in -1.:0.2:-0.1], [i for i in 0.1:0.2:10.]))
plotcor(Camh, [i for i in -1.:0.05:1.])
