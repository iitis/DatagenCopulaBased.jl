#!/usr/bin/env julia

using DatagenCopulaBased
using NPZ
using Distributions
using RandomNumbers.Xorshifts

## tests on simulated sata with custom rmg

c = GumbelCopula(2, 5.)
cg = NestedGumbelCopula([c], 1, 2.)

c = FrankCopula(2, 5.)
cf = NestedFrankCopula([c], 1, 2.)

c = ClaytonCopula(2, 5.)
cc = NestedClaytonCopula([c], 1, 2.)

c = AmhCopula(2, .9)
ca = NestedAmhCopula([c], 1, .3)

cch = ChainArchimedeanCopulas([-0.9, 3], "clayton")

cmo = MarshallOlkinCopula([1., 2., 3., 1., 2., 3., 4.])

cf = FrechetCopula(3, 0.5)

Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.]
ct = StudentCopula(Σ, 1)

t = 100_000

r = Xoroshiro128Plus()
x = simulate_copula(t, cc; rng = r)
npzwrite("copula.npy", x)

Σ = [1. 0.8 0.6 0.6; 0.8 1. 0.6 0.6; 0.6 0.6 1. 0.6; 0.6 0.6 0.6 1.]
x = Array(transpose(rand(MvNormal(Σ),t)))
y = gcop2arch(x, ["clayton"=>[1,2,3]])
npzwrite("transform.npy", y)
