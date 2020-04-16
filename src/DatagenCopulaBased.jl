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

  include("copulagendat.jl")

  include("eliptic_fr_mo_copulas.jl")
  include("archcopulagendat.jl")
  include("nestedarchcopulagendat.jl")
  include("chaincopulagendat.jl")

  include("corgen.jl")
  include("marshallolkincopcor.jl")
  include("archcopcorrelations.jl")

  include("subcopulasgendat.jl")

  export gaussian_cop, tstudent_cop, frechet, marshallolkin, gumbel, clayton, amh, frank
  export rev_gumbel, rev_clayton, rev_amh

  export archcopulagen, chaincopulagen, nestedarchcopulagen
  export cormatgen, cormatgen_constant, cormatgen_toeplitz, convertmarg!, gcop2arch
  export cormatgen_constant_noised, cormatgen_toeplitz_noised, cormatgen_rand
  export cormatgen_two_constant, cormatgen_two_constant_noised
  export chainfrechetcopulagen
  export gcop2tstudent, gcop2frechet, gcop2marshallolkin

  # obsolete implemntations
  export tstudentcopulagen, gausscopulagen, frechetcopulagen, marshallolkincopulagen
end
