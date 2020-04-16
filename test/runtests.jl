using Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase
using Random
using Distributed
using LinearAlgebra
using SharedArrays
using Combinatorics
using SpecialFunctions

import DatagenCopulaBased: simulate_copula
import DatagenCopulaBased: rand2cop, fncopulagen
import DatagenCopulaBased: logseriescdf, logseriesquantile, levyel, levygen, tiltedlevygen
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen
import DatagenCopulaBased: getV0, phi
import DatagenCopulaBased: testθ, useρ, useτ, testbivθ, usebivρ, getθ4arch
import DatagenCopulaBased: τ2λ, moρ2τ, norm2unifind
import DatagenCopulaBased: dilog, Debye, frankτ2θ, τ2θ, AMHτ2θ, Ccl, Cg
import DatagenCopulaBased: gumbelθ2ρ, claytonθ2ρ, gumbelρ2θ, claytonρ2θ, frankρ2θ, ρ2θ, AMHρ2θ
import DatagenCopulaBased: mocopula, arch_gen
import DatagenCopulaBased: nestedcopulag, testnestedθϕ, nestedstep
import DatagenCopulaBased: meanΣ, frechet, mean_outer, parameters, are_parameters_good, Σ_theor
import DatagenCopulaBased: getcors_advanced


include("tailtest.jl")
#include("univdatagentests.jl")

#include("eliptic_fr_mo_test.jl")
include("archcopulatests.jl")

#include("nestedarchcoptest.jl")
#include("subcopulastests.jl")
#include("chaincopulastests.jl")
#include("multiproctests.jl")
