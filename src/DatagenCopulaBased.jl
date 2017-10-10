module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using NLsolve
  using Combinatorics

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!, amhcopulagen, marshalolkincopulagen, copulamix
  export cormatgen, gumbelcopulagen, frankcopulagen, productcopula, copulamixbv, copulamixgenunif, copulamix1, claytoncopula, g2tsubcopula!
end
