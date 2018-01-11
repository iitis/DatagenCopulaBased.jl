module DatagenCopulaBased
  using Distributions
  using NLsolve
  using Combinatorics
  using PyCall
  using HypothesisTests
  using Cubature
  @pyimport scipy.cluster.hierarchy as sch

  include("sampleunivdists.jl")
  include("archcopcorrelations.jl")
  include("archcopulagendat.jl")
  include("nestedarchcopulagendat.jl")
  include("chaincopulagendat.jl")
  include("subcopulasgendat.jl")
  include("copulagendat.jl")
  include("marshalolkincopcorr.jl")

  export archcopulagen, chaincopulagen, nestedarchcopulagen
  export cormatgen, convertmarg!
  export tstudentcopulagen, gausscopulagen
  export frechetcopulagen, marshalolkincopulagen, chainfrechetcopulagen
  export copulamix
end
