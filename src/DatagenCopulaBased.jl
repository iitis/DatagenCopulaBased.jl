module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using QuadGK
  using NLsolve
  using Combinatorics

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")
  include("nestedcopula.jl")

  export tstudentcopulagen, gausscopulagen, convertmarg!, marshalolkincopulagen, archcopulagen
  export cormatgen, copulamixbv, g2tsubcopula!, copulamix, nestedgumbelcopula, frechetcopulagen, nestedclaytoncopula
  export nestedfrechetcopulagen, nestedamhcopula, nestedfrankcopula, frankgen
end
