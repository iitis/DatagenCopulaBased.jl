using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: lefttail, righttail, ρ2θ, AMHθ, rand2cop, g2tsubcopula!, logseriescdf, logseriesquantile, τ2λ₁₂

include("copulatests.jl")
