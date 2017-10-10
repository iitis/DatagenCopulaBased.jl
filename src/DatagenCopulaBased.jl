module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using NLsolve
  using Combinatorics

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!, amhcopulagen, marshalolkincopulagen, copulamix1
  export cormatgen, gumbelcopulagen, frankcopulagen, productcopula, copulamixgen, copulamixgenunif, copulamix1, claytoncopula, g2tsubcopula!
end
