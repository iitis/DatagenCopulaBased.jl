lefttail(v1::Vector{T}, v2::Vector{T}, α::T) where T <: AbstractFloat =
        sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)


 righttail(v1::Vector{T}, v2::Vector{T}, α::T) where T <: AbstractFloat =
         sum((v1 .> α) .* (v2 .> α))./(length(v1)*(1-α))
