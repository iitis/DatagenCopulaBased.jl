"""
    simulate_copula(t::Int, copula::Function, args...)

    returns t samples of the given copula function with args...

    supports following copulas:
        - gaussian_cop, args = Σ::Matrix{Float} (the correlation matrix, the covariance one will be normalised)
        - tstudent_cop,  args = Σ::Matrix{Float}, ν::Int
        - frechet, args = n::Int, α::Union{Int, Float} (number of marginals, parameter of maximal copula α ∈ [0,1])
        - frechet, args = n::Int, α::Union{Int, Float}, β::Union{Int, Float}, supported only for n = 2 and α, β, α+β ∈ [0,1]
        - marshalolkin, args = λ::Vector{Float64}
              n.o. margs = ceil(Int, log(2, length(λ)-1)), params: λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]
        - gumbel, clayton, amh (Ali-Mikhail-Haq), frank.  params: n::Int - n.o. margs, θ::Float - copula parameter, cor::String, keyword.
              if cor = ["Spearman", "Kendall"] uses these correlations in place of θ
        -rev_gumbel, rev_clayton, rev_amh - the same but the output is reversed: u →  1 .- u
"""

function simulate_copula(t::Int, copula::Function, args...; cor = "")
    if cor != ""
        return copula(t, args...; cor = cor)
    else
        return copula(t, args...)
    end
end


# Obsolete implemnetations

gausscopulagen(t::Int, Σ::Matrix{Float64}) = simulate_copula(t, gaussian_cop, Σ)

tstudentcopulagen(t::Int, Σ::Matrix{Float64}, ν::Int) = simulate_copula(t, tstudent_cop, Σ, ν)

frechetcopulagen(t::Int, args...) = simulate_copula(t, frechet, args...)

marshallolkincopulagen(t::Int, λ::Vector{Float64}) = simulate_copula(t, marshallolkin, λ)


function archcopulagen(t::Int, n::Int, θ::Union{Float64, Int}, copula::String;
                                                              rev::Bool = false,
                                                              cor::String = "")
    if copula == "gumbel"
        if !rev
            simulate_copula(t, gumbel, n, θ; cor = cor)
        else
            simulate_copula(t, rev_gumbel, n, θ; cor = cor)
        end
    elseif copula == "clayton"
        if !rev
            simulate_copula(t, clayton, n, θ; cor = cor)
        else
            simulate_copula(t, rev_clayton, n, θ; cor = cor)
        end
    elseif copula == "amh"
        if !rev
            simulate_copula(t, amh, n, θ; cor = cor)
        else
            simulate_copula(t, rev_amh, n, θ; cor = cor)
        end
    elseif copula == "frank"
        simulate_copula(t, frank, n, θ; cor = cor)
    else
        throw(AssertionError("$(copula) copula is not supported"))
    end
end
