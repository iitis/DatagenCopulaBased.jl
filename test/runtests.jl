using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: tail, ρ2θ, AMHθ, rand2cop, g2tsubcopula!, copulagen, g2tsubcopula!
import DatagenCopulaBased: logseriescdf, logseriesquantile, τ2λ, norm2unifind, makeind, τ2θ, AMHτ2θ
import DatagenCopulaBased: frankτ2θ, dilog, frankθ, levygen, Debye
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen

include("copulatests.jl")
include("bivariatecopulas.jl")
include("nestedctest.jl")
