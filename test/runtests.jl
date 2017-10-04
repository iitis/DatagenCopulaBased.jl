using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: lefttail, righttail, copuladeftest, claytonθ, g2clsubcopula, g2tsubcopula!

include("copulatests.jl")
