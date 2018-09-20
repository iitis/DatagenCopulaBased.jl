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


import DatagenCopulaBased: rand2cop, fncopulagen
import DatagenCopulaBased: logseriescdf, logseriesquantile, levyel, levygen, tiltedlevygen
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen
import DatagenCopulaBased: getV0, phi
import DatagenCopulaBased: testθ, useρ, useτ, testbivθ, usebivρ
import DatagenCopulaBased: τ2λ, moρ2τ, norm2unifind
import DatagenCopulaBased: dilog, Debye, frankτ2θ, τ2θ, AMHτ2θ, Ccl, Cg
import DatagenCopulaBased: gumbelθ2ρ, claytonθ2ρ, gumbelρ2θ, claytonρ2θ, frankρ2θ, ρ2θ, AMHρ2θ
import DatagenCopulaBased: mocopula, copulagen
import DatagenCopulaBased: nestedcopulag, testnestedθϕ, nestedstep
import DatagenCopulaBased: meanΣ, frechet, mean_outer, parameters, are_parameters_good, Σ_theor
import DatagenCopulaBased: getcors_advanced


include("tailtest.jl")
include("univdatagentests.jl")
include("archcopulatests.jl")
include("copulatests.jl")
include("nestedarchcoptest.jl")
include("subcopulastests.jl")
include("chaincopulastests.jl")
include("multiproctests.jl")
