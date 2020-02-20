#!/usr/bin/env julia

using DatagenCopulaBased
import DatagenCopulaBased: ρ2θ
using Distributions
using Cumulants
using PyPlot
using SymmetricTensors
using StatsBase
using PyCall
using JLD2
using FileIO
using Distributed

#using QuadGK
#using Cubature
#using NLsolve
#using LsqFit

mpl = pyimport("matplotlib")
mpl.rc("text", usetex=true)

addprocs(3)
@everywhere using DatagenCopulaBased
@everywhere using Cumulants
nprocs()


function ccl(u::Vector, θ::Union{Int, Float64})
  q = length(u)
  (1-q+sum(u.^(-θ)))^(-q-1/θ)*prod([u[j]^(-θ-1)*((j-1)*θ+1) for j in 1:q])
end

function munif(i::Vector{Int}, θ::Union{Int, Float64}, g::Function)
  n = length(i)
  f(x) = mapreduce(j -> x[j], *, i)*g(x, θ)
  hcubature(f, [fill(0, n)...],[fill(1, n)...])[1]
end

function c3(θ::Union{Int, Float64}, g::Function)
 c3ofd = munif([1,2,3], θ, g)-3*munif([1,2], θ, g)*munif([1], θ, g)+2*munif([1], θ, g)^3
 c3mix = munif([1,1,2], θ, g)-munif([1,1], θ, g)*munif([1], θ, g)-2*munif([1,2], θ, g)*munif([1], θ, g)+2*munif([1], θ, g)^3
 c3diag = munif([1,1,1], θ, g)-3*munif([1,1], θ, g)*munif([1], θ, g)+2*munif([1], θ, g)^3
 [c3diag, c3mix, c3ofd]
end


function mmarg(i::Vector{Int}, θ::Union{Int, Float64}, g::Function, d)
  n = length(i)
  f(x) = mapreduce(j -> quantile.(d, x[j]), *, i)*g(x, θ)
  hcubature(f, [fill(0.00001, n)...],[fill(0.999999, n)...])[1]
end

c3symmarg(θ::Union{Int, Float64}, g::Function, d) =[mmarg([1,1,1], θ, g, d), mmarg([1,1,2], θ, g, d), mmarg([1,2,3], θ, g, d)]

function c4symmarg(θ::Union{Int, Float64}, g::Function, d)
  a = mmarg([1,1,1,1], θ, g, d) - 3*mmarg([1,1], θ, g, d)^2
  println("a")
  b = mmarg([1,1,1,2], θ, g, d) - 3*mmarg([1,1], θ, g, d)*mmarg([1,2], θ, g, d)
  println("b")
  c = mmarg([1,1,2,2], θ, g, d) - mmarg([1,1], θ, g, d)^2 - 2*mmarg([1,2], θ, g, d)^2
  println("c")
  e = mmarg([1,1,2,3], θ, g, d) - mmarg([1,1], θ, g, d)*mmarg([1,2], θ, g, d) - 2*mmarg([1,2], θ, g, d)^2
  println("d")
  f = mmarg([1,2,3,4], θ, g, d) - 3*mmarg([1,2], θ, g, d)^2
  [a,b,c,e,f]
end

function c3empirical(θ::Union{Float64, Int}, cop::String, d, rev = false, t::Int = 1_000_000)
  u = quantile.(d, archcopulagen(t, 3, θ, cop; rev = rev))
  c = cumulants(u, 3)[3]
  println(θ)
  [c[1,1,1], c[1,1,2], c[1,2,3]]
end

function c4empirical(θ::Union{Float64, Int}, cop::String, d, rev::Bool= false, t::Int = 5000000)
  u = quantile.(d, archcopulagen(t, 4, θ, cop; rev = rev))
  c = cumulants(u, 4)[4]
  [c[1,1,1,1], c[1,1,1,2], c[1,1,2,2], c[1,1,2,3], c[1,2,3,4]]
end

function stats3theoreunifcl()
  θ = map(i -> ρ2θ(i, "clayton"), 0.05:0.05:.95)
  n = length(θ)
  t = zeros(n, 3)
  for i in 1:n
    t[i,:] = c3(θ[i], ccl)
  end
  t
end


function statstheoreticalclayton(m::Int = 3, d = Normal(0,1))
  θ = map(i -> ρ2θ(i, "clayton"), 0.05:0.05:.95)
  n = length(θ)
  t = (m == 3) ? zeros(n, 3) : zeros(n, 5)
  for i in 1:n
    t[i,:] = (m == 3) ? c3symmarg(θ[i], ccl, d) : c4symmarg(θ[i], ccl, d)
    println(i)
  end
  t
end

#save("c3clgmarg.jld2", "emp", e, "theoret", t, "rho", ρ)

function empiricalcums(copula::String, m::Int, ρ::Vector{Float64}, rev = false)
  θ = map(i -> ρ2θ(i, copula), ρ)
  println(θ)
  n = length(θ)
  e = (m == 3) ? zeros(n, 3) : zeros(n, 5)
  for i in 1:n
    println(i)
    e[i,:] = (m == 3) ? c3empirical(θ[i], copula, Normal(0,1), rev) : c4empirical(θ[i], copula, Normal(0,1), rev)
  end
  e
end


function plotc3empth(e,t, ρ)
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  fx = matplotlib.ticker.ScalarFormatter()
  fx.set_powerlimits((-1, 4))
  ax.yaxis.set_major_formatter(fx)
  plot(ρ, e[:,3], "o", color = "black", label = "\$ \\mathbf{i} = (1,2,3) \$", markersize = 1.)
  plot(ρ, t[:,3], color = "black", linewidth = 0.5)
  plot(ρ, e[:,2], "o", color = "blue", label = "\$ \\mathbf{i} = (1,1,2) \$", markersize = 1.)
  plot(ρ, t[:,2], color = "blue", linewidth = 0.5)
  plot(ρ, e[:,1], "o", color = "red", label = "\$ \\mathbf{i} = (1,1,1) \$", markersize = 1.)
  plot(ρ, t[:,1], color = "red", linewidth = 0.5)
  PyPlot.ylabel("cumulant element", labelpad = -1.0)
  PyPlot.xlabel("Spearman \$ \\rho \$ between marginals", labelpad = -1.0)
  ax.legend(fontsize = 6, loc = 3, ncol = 1)
  subplots_adjust(left = 0.12, bottom = 0.12)
  fig.savefig("pics/c3clgmarg.pdf")
end


function plotc3emp(e, ρ, cop::String)
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  fx = matplotlib.ticker.ScalarFormatter()
  fx.set_powerlimits((-1, 4))
  ax.yaxis.set_major_formatter(fx)
  plot(ρ, e[:,3], "o", color = "black", label = "\$ \\mathbf{i} = (1, 2, 3) \$", markersize = 1.)
  plot(ρ, e[:,2], "o", color = "blue", label = "\$ \\mathbf{i} = (1, 1, 2) \$", markersize = 1.)
  plot(ρ, e[:,1], "o", color = "red", label = "\$ \\mathbf{i} = (1, 1, 1) \$", markersize = 1.)
  PyPlot.ylabel("cumulant element", labelpad = -1.0)
  PyPlot.xlabel("Spearman \$ \\rho \$ between marginals", labelpad = -1.0)
  #ax.legend(fontsize = 7, loc = 3, ncol = 1)
  subplots_adjust(left = 0.16, bottom = 0.12)
  fig.savefig("pics/c3"*cop*"gmarg.pdf")
end


function plotc4emp(e, ρ, cop::String)
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  fx = matplotlib.ticker.ScalarFormatter()
  fx.set_powerlimits((-1, 4))
  ax.yaxis.set_major_formatter(fx)
  plot(ρ, e[:,5], "o", color = "black", label = "\$ i_1, i_2, i_3, i_4 \$", markersize = 1.)
  plot(ρ, e[:,4], "o", color = "brown", label = "\$ i_1, i_1, i_2, i_3 \$", markersize = 1.)
  plot(ρ, e[:,3], "o", color = "gray", label = "\$ i_1, i_1, i_2, i_2 \$", markersize = 1.)
  plot(ρ, e[:,2], "o", color = "blue", label = "\$ i_1, i_1, i_1, i_2 \$", markersize = 1.)
  plot(ρ, e[:,1], "o", color = "red", label = "\$ i_1, i_1, i_1, i_1 \$", markersize = 1.)
  PyPlot.ylabel("cumulant element", labelpad = -1.0)
  PyPlot.xlabel("Spearman \$ \\rho \$", labelpad = -1.0)
  #ax.legend(fontsize = 4.5, loc = 3, ncol = 1)
  subplots_adjust(left = 0.16, bottom = 0.12)
  fig.savefig("pics/c4"*cop*"gmarg.pdf")
end

function plotc4empth(e,t, ρ, ρ1)
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  fx = matplotlib.ticker.ScalarFormatter()
  fx.set_powerlimits((-1, 4))
  ax.yaxis.set_major_formatter(fx)
  plot(ρ1, t[:,5], color = "black", linewidth = 0.5)
  plot(ρ, e[:,5], "o", color = "black", label = "\$ i_1, i_2, i_3, i_4 \$", markersize = 1.)
  plot(ρ1, t[:,4], color = "brown", linewidth = 0.5)
  plot(ρ, e[:,4], "o", color = "brown", label = "\$ i_1, i_1, i_2, i_3 \$", markersize = 1.)
  plot(ρ, e[:,3], "o", color = "gray", label = "\$ i_1, i_1, i_2, i_2 \$", markersize = 1.)
  plot(ρ1, t[:,3], color = "gray", linewidth = 0.5)
  plot(ρ1, t[:,2], color = "blue", linewidth = 0.5)
  plot(ρ, e[:,2], "o", color = "blue", label = "\$ i_1, i_1, i_1, i_2 \$", markersize = 1.)
  plot(ρ, e[:,1], "o", color = "red", label = "\$ i_1, i_1, i_1, i_1 \$", markersize = 1.)
  plot(ρ1, t[:,1], color = "red", linewidth = 0.5)
  PyPlot.ylabel("cumulant element", labelpad = -1.0)
  PyPlot.xlabel("Spearman \$ \\rho \$", labelpad = -1.0)
  ax.legend(fontsize = 4.5, loc = 3, ncol = 1)
  subplots_adjust(left = 0.12, bottom = 0.12)
  fig.savefig("pics/c4clgmarg.pdf")
end


if false
  ρ = collect(0.05:0.05:.95)
  e4 = empiricalcums("gumbel", 4, ρ, false)
  save("pics/c4gugmarg.jld2", "emp", e4,"rho", ρ)
  t = statstheoreticalclayton(4, Normal(0,1));
  save("pics/teorc4clgmarg.jld2", "theor", t,"rho", ρ)

  ρ = [i for i in 0.02:0.02:.85]
  e = empiricalcums("frank", 3, ρ)
  save("pics/c3frgmarg.jld2", "emp", e,"rho", ρ)

  ρ = [i for i in 0.02:0.02:.98]
  e = empiricalcums("gumbel", 3, ρ)
  save("pics/c3gugmarg.jld2", "emp", e,"rho", ρ)

  ρ = [i for i in 0.02:0.02:.5]
  e = empiricalcums("amh", 3, ρ)
  save("pics/c3amhgmarg.jld2", "emp", e,"rho", ρ)
end


c3els = load("pics/c3clgmarg.jld2")
plotc3empth(c3els["emp"],c3els["theoret"], c3els["rho"])

c3els = load("pics/c3gugmarg.jld2")
plotc3emp(c3els["emp"], c3els["rho"], "gu")

c3els = load("pics/c3amhgmarg.jld2")
plotc3emp(c3els["emp"], c3els["rho"], "amh")

c4els = load("pics/c4clgmarg.jld2")
plotc4emp(c4els["emp"], c4els["rho"], "cl")

emp = load("pics/c4clgmarg.jld2")
t = load("pics/teorc4clgmarg.jld2")
plotc4empth(emp["emp"], t["theor"], emp["rho"], t["rho"])
