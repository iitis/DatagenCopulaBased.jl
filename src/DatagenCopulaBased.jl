module DatagenCopulaBased
  using HypothesisTests
  using Distributions

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export clcopulagen, tcopulagen, gcopulagen, convertmarg!
  export subcopdatagen, cormatgen, g2clsubcopula, g2tsubcopula!, clcopulagenapprox
end
