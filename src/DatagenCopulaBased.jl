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
  if VERSION >= v"1.3"
    using CompilerSupportLibraries_jll
  end

  include("simulate_copula.jl")
  include("sampleunivdists.jl")
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


  export GaussianCopula, StudentCopula, FrechetCopula, MarshallOlkinCopula
  export GumbelCopula, GumbelCopulaRev, ClaytonCopula, ClaytonCopulaRev, AmhCopula, AmhCopulaRev, FrankCopula
  export NestedClaytonCopula, NestedAmhCopula, NestedFrankCopula, NestedGumbelCopula
  export DoubleNestedGumbelCopula, HierarchicalGumbelCopula, NestedGumbelCopula
  export ChainArchimedeanCopulas, ChainFrechetCopulas
  export SpearmanCorrelation, KendallCorrelation, CorrelationType
  export cormatgen, cormatgen_constant, cormatgen_toeplitz, convertmarg!
  export cormatgen_constant_noised, cormatgen_toeplitz_noised, cormatgen_rand
  export cormatgen_two_constant, cormatgen_two_constant_noised

  export gcop2tstudent, gcop2frechet, gcop2marshallolkin, gcop2arch

  export simulate_copula, simulate_copula!

end
