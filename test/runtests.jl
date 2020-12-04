if VERSION >= v"1.3"
  using CompilerSupportLibraries_jll
end
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
using StableRNGs


import DatagenCopulaBased: rand2cop, fncopulagen
#import chainfrechetcopulagen
import DatagenCopulaBased: logseriescdf, levyel, tiltedlevygen
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen
import DatagenCopulaBased: testθ, useρ, useτ, testbivθ, usebivρ, getθ4arch
import DatagenCopulaBased: τ2λ, moρ2τ, norm2unifind
import DatagenCopulaBased: dilog, Debye, frankτ2θ, τ2θ, AMHτ2θ, Ccl, Cg
import DatagenCopulaBased: gumbelθ2ρ, claytonθ2ρ, gumbelρ2θ, claytonρ2θ, frankρ2θ, ρ2θ, AMHρ2θ
import DatagenCopulaBased: mocopula, arch_gen
import DatagenCopulaBased: nestedcopulag
import DatagenCopulaBased: meanΣ, frechet, mean_outer, parameters, are_parameters_good, Σ_theor
import DatagenCopulaBased: getcors_advanced
import DatagenCopulaBased: random_unit_vector
import DatagenCopulaBased: frechet_el!, frechet_el2!, mocopula_el
#import DatagenCopulaBased: archcopulagen, chaincopulagen, nestedarchcopulagen

# axiliary tests
include("tailtest.jl")
include("univdatagentests.jl")
include("marg_cor_tests.jl")

# tests particular copulas generators
include("eliptic_fr_mo_test.jl")
include("archcopulatests.jl")
include("nestedarchcoptest.jl")
include("chaincopulastests.jl")

# test transforming of data by introducing higher order corss-correlations
include("higher_order_cors_tests.jl")
