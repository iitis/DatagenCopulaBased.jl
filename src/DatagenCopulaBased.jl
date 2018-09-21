module DatagenCopulaBased
  using Distributions
  using Combinatorics
  using HypothesisTests
  using HCubature
  using StatsBase
  using QuadGK
  using Roots
  using SpecialFunctions
  using Random
  using LinearAlgebra
  using Distributed
  using SharedArrays


  include("sampleunivdists.jl")
  include("archcopcorrelations.jl")
  include("archcopulagendat.jl")
  include("nestedarchcopulagendat.jl")
  include("chaincopulagendat.jl")
  include("subcopulasgendat.jl")
  include("copulagendat.jl")
  include("marshallolkincopcor.jl")

  export archcopulagen, chaincopulagen, nestedarchcopulagen
  export cormatgen, cormatgen_constant, cormatgen_toeplitz, convertmarg!, gcop2arch
  export cormatgen_constant_noised, cormatgen_toeplitz_noised, cormatgen_rand
  export cormatgen_two_constant, cormatgen_two_constant_noised
  export tstudentcopulagen, gausscopulagen
  export frechetcopulagen, marshallolkincopulagen, chainfrechetcopulagen
  export gcop2tstudent, gcop2frechet, gcop2marshallolkin
end
