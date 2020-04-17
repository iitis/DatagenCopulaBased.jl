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

  # dispatching of the generator
  include("copulagendat.jl")

  # axiliary function for correlqations
  include("corgen.jl")
  include("marshallolkincopcor.jl")
  include("archcopcorrelations.jl")

  # particular copulas famillies
  include("eliptic_fr_mo_copulas.jl")
  include("archcopulagendat.jl")
  include("nestedarchcopulagendat.jl")
  include("chaincopulagendat.jl")

  # change Gaussian data by adding higher order cross-correlations
  include("add_higher_order_cors.jl")

  export gaussian_cop, tstudent_cop, frechet, marshallolkin, gumbel, clayton, amh, frank
  export rev_gumbel, rev_clayton, rev_amh

  export archcopulagen, chaincopulagen, nestedarchcopulagen
  export cormatgen, cormatgen_constant, cormatgen_toeplitz, convertmarg!, gcop2arch
  export cormatgen_constant_noised, cormatgen_toeplitz_noised, cormatgen_rand
  export cormatgen_two_constant, cormatgen_two_constant_noised
  export chainfrechetcopulagen
  export gcop2tstudent, gcop2frechet, gcop2marshallolkin
  export nested_gumbel, nested_clayton, nested_frank, nested_amh
  export chain_frechet, chain_archimedeans, rev_chain_archimedeans

  # obsolete implemntations
  export tstudentcopulagen, gausscopulagen, frechetcopulagen, marshallolkincopulagen
end
