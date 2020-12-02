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
  #include("copulagendat.jl")

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


  export Gaussian_cop, Student_cop, Frechet_cop, Marshall_Olkin_cop
  export Gumbel_cop, Gumbel_cop_rev, Clayton_cop, Clayton_cop_rev, AMH_cop, AMH_cop_rev, Frank_cop
  export Nested_Clayton_cop, Nested_AMH_cop, Nested_Frank_cop, Nested_Gumbel_cop
  export Double_Nested_Gumbel_cop, Hierarchical_Gumbel_cop
  export Chain_of_Archimedeans, Chain_of_Frechet
  export SpearmanCorrelation, KendallCorrelation, CorrelationType
  export cormatgen, cormatgen_constant, cormatgen_toeplitz, convertmarg!
  export cormatgen_constant_noised, cormatgen_toeplitz_noised, cormatgen_rand
  export cormatgen_two_constant, cormatgen_two_constant_noised

  export gcop2tstudent, gcop2frechet, gcop2marshallolkin, gcop2arch

  export simulate_copula, simulate_copula!

  # obsolete implemntations
  #export tstudentcopulagen, gausscopulagen, frechetcopulagen, marshallolkincopulagen
end
