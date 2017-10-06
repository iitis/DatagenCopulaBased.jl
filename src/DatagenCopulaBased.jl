module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using NLsolve
  using PyCall
  using Combinatorics
  @pyimport numpy.random as npr

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!, amhcopulagen, marshalolkincopulagen
  export cormatgen, gumbelcopulagen, frankcopulagen, productcopula, copulamixgen, copulamixgenunif
end
