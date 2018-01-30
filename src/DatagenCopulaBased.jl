module DatagenCopulaBased
  using Distributions
  using Combinatorics
  using HypothesisTests
  using Cubature
  using StatsBase
  using QuadGK
  using Roots
  using Iterators


  include("sampleunivdists.jl")
  include("archcopcorrelations.jl")
  include("archcopulagendat.jl")
  include("nestedarchcopulagendat.jl")
  include("chaincopulagendat.jl")
  include("subcopulasgendat.jl")
  include("copulagendat.jl")
  include("marshalolkincopcor.jl")

  export archcopulagen, chaincopulagen, nestedarchcopulagen
  export cormatgen, convertmarg!, gcop2arch
  export tstudentcopulagen, gausscopulagen
  export frechetcopulagen, marshalolkincopulagen, chainfrechetcopulagen
  export copulamix
end
