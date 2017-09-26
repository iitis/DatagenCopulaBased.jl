module DatagenCopulaBased

  using Distributions

  include("copulagendat.jl")
  include("subcopgendat.jl")
  include("helpers.jl")

  export clcopulagen, tcopulagen, gcopulagen, convertmarg!
  export subcopdatagen, covmatgen, g2clsubcopula, g2tsubcopula!
end
