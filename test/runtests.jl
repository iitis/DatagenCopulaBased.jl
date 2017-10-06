using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase
using PyCall
@pyimport numpy.random as npr

import DatagenCopulaBased: lefttail, righttail, claytonθ, rand2cop, g2tsubcopula!

include("copulatests.jl")
