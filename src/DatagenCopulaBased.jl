module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using QuadGK
  using NLsolve
  using Combinatorics
  using PyCall
  @pyimport scipy.cluster.hierarchy as sch
  
  include("gendat.jl")
  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")
  include("nestedcopula.jl")

  export tstudentcopulagen, gausscopulagen, frechetcopulagen, marshalolkincopulagen, archcopulagen
  export cormatgen, copulamixbv, copulamix, convertmarg!
  export nestedfrechetcopulagen, nestedarchcopulagen, nestedgumbelcopula
end
