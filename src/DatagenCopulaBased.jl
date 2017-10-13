module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using NLsolve
  using Combinatorics

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")
  include("nastedcopula.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!, amhcopulagen, marshalolkincopulagen, copulamix
  export cormatgen, gumbelcopulagen, frankcopulagen, productcopula, copulamixbv, g2tsubcopula!
  export nastedgumbelcopula
end
