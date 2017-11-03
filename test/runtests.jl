using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: tail, ρ2θ, AMHθ, rand2cop, g2tsubcopula!, copulagen, g2tsubcopula!
import DatagenCopulaBased: logseriescdf, logseriesquantile, levyel, levygen, tiltedlevygen
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen
import DatagenCopulaBased: frankτ2θ, dilog, frankθ, Debye, τ2λ, norm2unifind, makeind, τ2θ, AMHτ2θ


include("datagentests.jl")
include("copulatests.jl")
#include("bivariatecopulas.jl")
#include("nestedctest.jl")
