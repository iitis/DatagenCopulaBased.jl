using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: tail, ρ2θ, AMHθ, rand2cop, g2tsubcopula!, copulagen
import DatagenCopulaBased: logseriescdf, logseriesquantile, τ2λ, norm2unifind, makeind, τ2θ

include("bivariatecopulas.jl")
include("copulatests.jl")
include("nastedcopulas.jl")
