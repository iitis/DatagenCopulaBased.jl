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
  export cormatgen, cormatgen_constant, cormatgen_toeplitz, convertmarg!, gcop2arch
  export cormatgen_constant_noised, cormatgen_toeplitz_noised
  export tstudentcopulagen, gausscopulagen
  export frechetcopulagen, marshalolkincopulagen, chainfrechetcopulagen
  export copulamix
end
