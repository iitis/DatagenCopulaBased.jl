module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using QuadGK
  using NLsolve
  using Combinatorics

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")
  include("nastedcopula.jl")

  export tstudentcopulagen, gausscopulagen, convertmarg!, marshalolkincopulagen, archcopulagen
  export cormatgen, copulamixbv, g2tsubcopula!, copulamix, nastedgumbelcopula, frechetcopulagen
  export nastedfrechetcopulagen, nastedamhcopula
end
