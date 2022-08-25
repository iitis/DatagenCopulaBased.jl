abstract type Copula{T} end 
value_eltype(::Copula{T}) where T = T



function simulate_copula(t, copula; rng = Random.GLOBAL_RNG)
    U = zeros(value_eltype(copula), t, copula.n)
    simulate_copula!(U, copula; rng = rng)
    return U
end