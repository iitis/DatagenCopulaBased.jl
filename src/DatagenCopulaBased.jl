module DatagenCopulaBased
  using Distributions
  using QuadGK
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


  export tstudentcopulagen, gausscopulagen, frechetcopulagen, marshalolkincopulagen
  export archcopulagen, chaincopulagen
  export cormatgen, copulamix, convertmarg!
  export chainfrechetcopulagen, nestedarchcopulagen
end
