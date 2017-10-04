module DatagenCopulaBased
  using HypothesisTests
  using Distributions
  using QuadGK
  using NLsolve
  using PyCall
  @pyimport numpy.random as npr

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!, amhcopulagen
  export subcopdatagen, cormatgen, gumbelcopulagen, frankcopulagen, productcopula
end
