using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: rand2cop, bivariatecopulamix
import DatagenCopulaBased: logseriescdf, logseriesquantile, levyel, levygen, tiltedlevygen
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen
import DatagenCopulaBased: getV0, phi
import DatagenCopulaBased: testθ, useρ, useτ, tail, ρ2θ, AMHθ, testbivθ, usebivρ
import DatagenCopulaBased: frankτ2θ, dilog, frankθ, Debye, τ2λ, norm2unifind, makeind, τ2θ, AMHτ2θ
import DatagenCopulaBased: mocopula, copulagen
import DatagenCopulaBased: g2tsubcopula!, nestedcopulag, testnestedθϕ, nestedstep

include("datagentests.jl")
include("copulatests.jl")
include("bivcoptests.jl")
include("nestedctest.jl")
