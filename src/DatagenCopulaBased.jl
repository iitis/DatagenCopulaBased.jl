module DatagenCopulaBased

  using Distributions

  include("copulagendat.jl")
  include("subcopgendat.jl")

  export subcopdatagen, clcopulagen, cormatgen, tcopulagen, gcopulagen,
   gcopulatmarg, tcopulagmarg, tdistdat, normdist

end
