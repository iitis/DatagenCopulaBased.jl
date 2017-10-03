lefttail(v1::Vector{T}, v2::Vector{T}, α::T = 0.001) where T <: AbstractFloat =
        sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)


 righttail(v1::Vector{T}, v2::Vector{T}, α::T = 0.999) where T <: AbstractFloat =
         sum((v1 .> α) .* (v2 .> α))./(length(v1)*(1-α))


function copuladeftest(v1::Vector{T}, v2::Vector{T}, α::Vector{T}, β::Vector{T}) where T <: AbstractFloat
 sum((v1 .> α[1]) .* (v2 .> β[1]))+ sum((v1 .> α[2]) .* (v2 .> β[2])) - sum((v1 .> α[1]) .* (v2 .> β[2]))+ sum((v1 .> α[2]) .* (v2 .> β[1]))
end

claytonθ(ρ::Union{Float64, Int}) = 4*asin(ρ)/(pi-2*asin(ρ))

gumbelθ(ρ::Union{Float64, Int}) = 1/(1-2*asin(ρ)/pi)

claytonθ2ρ(θ::Union{Float64, Int}) = sin(θ*pi/(4+2*θ))

# for frank copula

D(θ) = 1/θ*(quadgk(i -> i/(exp(i)-1), 0, θ)[1])

frankτ(θ) = 1+4*(D(θ)-1)/θ

frankρ(θ) = sin(pi*frankτ(θ)/2)

# approx

Frankθ2ρ(θ) = 0.69*atan.(0.28*θ)

Frankθ(ρ) = 1/0.28*tan(ρ/0.69)
