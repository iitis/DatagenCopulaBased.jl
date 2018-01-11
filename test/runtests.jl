using Base.Test
using DatagenCopulaBased
using Distributions
using HypothesisTests
using StatsBase

import DatagenCopulaBased: rand2cop, bivariatecopulamix, fncopulagen
import DatagenCopulaBased: logseriescdf, logseriesquantile, levyel, levygen, tiltedlevygen
import DatagenCopulaBased: Ginv, InvlaJ, sampleInvlaJ, elInvlaF, nestedfrankgen
import DatagenCopulaBased: getV0, phi
import DatagenCopulaBased: testθ, useρ, useτ, ρ2θ, AMHθ, testbivθ, usebivρ
import DatagenCopulaBased: frankτ2θ, dilog, frankθ, Debye, τ2λ, norm2unifind, makeind, τ2θ, AMHτ2θ
import DatagenCopulaBased: mocopula, copulagen
import DatagenCopulaBased: g2tsubcopula!, nestedcopulag, testnestedθϕ, nestedstep
import DatagenCopulaBased: findsimilar, getclust

include("tailtest.jl")
include("archcopulatests.jl")
include("chaincopulastests.jl")
include("nestedarchcoptest.jl")
include("univdatagentests.jl")
include("subcopulastests.jl")
include("copulatests.jl")
