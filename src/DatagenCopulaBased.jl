module DatagenCopulaBased
  using HypothesisTests
  using Distributions

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!
  export subcopdatagen, cormatgen, claytonsubcopulagen, revclaytoncopulagen
  export revclaytonsubcopulagen, gumbelcopulagen
  export productcopula, minimalcopula, maximalcopula, mixedcopula
end
