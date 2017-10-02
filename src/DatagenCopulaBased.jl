module DatagenCopulaBased
  using HypothesisTests
  using Distributions

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export claytoncopulagen, tstudentcopulagen, gausscopulagen, convertmarg!
  export subcopdatagen, cormatgen, gumbelcopulagen, frankcopulagen, productcopula
end
