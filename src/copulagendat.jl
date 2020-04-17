"""
    simulate_copula(t::Int, copula::Function, args...)

    returns t samples of the given copula function with args...

    supports following copulas:
        elliptical
        - gaussian_cop,
                args: Σ::Matrix{Float} (the correlation matrix, the covariance one will be normalised)
        - tstudent_cop,
                args: Σ::Matrix{Float}, ν::Int

        - frechet,
            args: n::Int, α::Union{Int, Float} (number of marginals, parameter of maximal copula α ∈ [0,1])
             or
            args: n::Int, α::Union{Int, Float}, β::Union{Int, Float}, supported only for n = 2 and α, β, α+β ∈ [0,1]

        - chain_frechet, simulates a chain of bivariate frechet copulas
            args: α::Vector{Float64}, β::Vector{Float64} = zero(α) - vectors of
            parameters of subsequent bivariate copulas. Each element must fulfill
            conditions of the frechet copula. n.o. marginals = length(α)+1.
            We require length(α) = length(β).


        - marshalolkin,
            args: λ::Vector{Float64} λ = [λ₁, λ₂, ..., λₙ, λ₁₂, λ₁₃, ..., λ₁ₙ, λ₂₃, ..., λₙ₋₁ₙ, λ₁₂₃, ..., λ₁₂...ₙ]
              n.o. margs = ceil(Int, log(2, length(λ)-1)), params:

        - archimedean: gumbel, clayton, amh (Ali-Mikhail-Haq), frank.
            args: n::Int - n.o. margs, θ::Float - copula parameter, cor::String, keyword.
              if cor = ["Spearman", "Kendall"] uses these correlations in place of θ
        - rev_gumbel, rev_clayton, rev_amh - the same but the output is reversed: u →  1 .- u

        - nested archimedean
            - nested_gumbel nested_clayton, nested_amh, nested_frank
              args:  - n::Vector{Int}, ϕ::Vector{Float64} (sizes and params of children copulas)
                     - θ::Float64, m::Int = 0 (param and additional size of parent copula)
        - double nested (only Gumbel)
            - nested_gumbel
             args:  - n::Vector{Vector{Int}} (sizes of ground children copulas)
                    - Ψ::Vector{Vector{Float64}}, ϕ::Vector{Float64}, θ::Float64 (params of ground childeren, children and parent copulas)
        - hierarchical nested (only Gumbel)
            - nested_gumbel
                args: θ::Vector{Float64} - vector of parameters from ground ground child to parent
                                    all copuals are bivariate n = length(θ)+1

        - chain_archimedeans  simulate the chain of bivariate archimedean copula,

            args: - θ::Union{Vector{Float64}, Vector{Int}} - parameters of subsequent bivariate copulas
                  - copula::Union{Vector{String}, String}, indicates a bivariate copulas
                        or their sequence, supported are supports: clayton, frank and amh famillies
                 - keyword cor, if cor = ["Spearman", "Kendall"] uses these correlations in place of parameters
                        of subsequent buvariate copulas
        - rev_chain_archimedeans reversed version of the chain_archimedeans

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


function nestedarchcopulagen(t::Int, n::Vector{Int}, ϕ::Vector{Float64}, θ::Float64, copula::String, m::Int = 0)
    if copula == "gumbel"
        simulate_copula(t, nested_gumbel, n, ϕ, θ, m)
    elseif copula == "clayton"
        simulate_copula(t, nested_clayton, n, ϕ, θ, m)
    elseif copula == "amh"
        simulate_copula(t, nested_amh, n, ϕ, θ, m)
    elseif copula == "frank"
        simulate_copula(t, nested_frank, n, ϕ, θ, m)
    else
        throw(AssertionError("$(copula) copula is not supported"))
    end
end


function nestedarchcopulagen(t::Int, n::Vector{Vector{Int}}, Ψ::Vector{Vector{Float64}},
                                                             ϕ::Vector{Float64}, θ::Float64,
                                                             copula::String = "gumbel")
  copula == "gumbel" || throw(AssertionError("generator supported only for gumbel familly"))
  simulate_copula(t, nested_gumbel, n, Ψ, ϕ, θ)
end


function nestedarchcopulagen(t::Int, θ::Vector{Float64}, copula::String = "gumbel")
    copula == "gumbel" || throw(AssertionError("generator supported only for gumbel familly"))
    simulate_copula(t, nested_gumbel, θ)
end


function chainfrechetcopulagen(t::Int, α::Vector{Float64}, β::Vector{Float64} = zero(α))
    simulate_copula(t, chain_frechet, α, β)
end

VFI = Union{Vector{Float64}, Vector{Int}}

function chaincopulagen(t::Int, θ::VFI, copula::Union{Vector{String}, String};
                                        rev::Bool = false, cor::String = "")
    if rev == false
      return simulate_copula(t, chain_archimedeans, θ, copula, cor = cor)
    else
      return simulate_copula(t, rev_chain_archimedeans, θ, copula, cor = cor)
    end
 end
