#!/usr/bin/env julia

using DatagenCopulaBased
using NPZ
using Distributions
using RandomNumbers.Xorshifts

## tests on simulated sata with custom rmg

c = Gumbel_cop(2, 5.)
cg = Nested_Gumbel_cop([c], 1, 2.)

c = Frank_cop(2, 5.)
cf = Nested_Frank_cop([c], 1, 2.)

c = Clayton_cop(2, 5.)
cc = Nested_Clayton_cop([c], 1, 2.)

c = AMH_cop(2, .9)
ca = Nested_AMH_cop([c], 1, .3)

cch = Chain_of_Archimedeans([-0.9, 3], "clayton")

cmo = Marshall_Olkin_cop([1., 2., 3., 1., 2., 3., 4.])

cf = Frechet_cop(3, 0.5)

Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.]
ct = Student_cop(Σ, 1)

t = 100_000

r = Xoroshiro128Plus()
x = simulate_copula(t, cch; rng = r)
npzwrite("copula.npy", x)

Σ = [1. 0.8 0.6 0.6; 0.8 1. 0.6 0.6; 0.6 0.6 1. 0.6; 0.6 0.6 0.6 1.]
x = Array(transpose(rand(MvNormal(Σ),t)))
y = gcop2arch(x, ["gumbel"=>[1,2,3]])
npzwrite("transform.npy", y)
