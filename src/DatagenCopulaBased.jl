module DatagenCopulaBased
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

  export tstudentcopulagen, gausscopulagen, frechetcopulagen, marshalolkincopulagen
  export archcopulagen, bivariatecopgen
  export cormatgen, copulamix, convertmarg!
  export bivfrechetcopulagen, nestedarchcopulagen
end
