using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests

import DatagenCopulaBased: invers_gen, lefttail, righttail, copuladeftest, claytonθ, g2clsubcopula, g2tsubcopula!

include("copulatests.jl")
