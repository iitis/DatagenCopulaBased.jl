"""
  function tail(v1::Vector{Float}, v2::Vector{Float}, α::Float = 0.002, tail::String)


Returns empirical left and right tail dependency of bivariate data
"""

function tail(v1::Vector{T}, v2::Vector{T}, tail::String, α::T = 0.002) where T <: AbstractFloat
  if tail == "l"
    return sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)
  elseif tail == "r"
    return sum((v1 .> (1-α)) .* (v2 .> (1-α)))./(length(v1)*α)
  end
  0.
end


@testset "tail dependencies test" begin
  v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
  v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
  @test tail(v1, v2,  "l", 0.1) ≈ 0.5
  @test tail(v1, v2, "r", 0.1) ≈ 0.5
end
