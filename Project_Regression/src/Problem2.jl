using Plots
function noisySin(N::Integer;σ²=0.09)

    # 1. Generate X_n: Drawn from Uniform distribution U(0, 1)
    # rand(N) produces N samples uniformly distributed between 0 and 1.
    X = rand(N)
    
    # 2. Calculate the Deterministic Sinusoidal Term: sin(2πX_n)
    # The term 2π is calculated by 2 * pi
    sin_term = sin.(2 * pi .* X)
    
    # 3. Generate the Noise Term: N(0, 0.09)
    # The variance (σ²) is 0.09, so the standard deviation (σ) is sqrt(0.09) = 0.3
    noise_std = sqrt(0.09)
    
    # randn(N) produces N samples from N(0, 1). We scale it by noise_std (0.3).
    noise_vector = randn(N) .* noise_std
    
    # 4. Compute the Final Target Value: Y_n = sin(2πX_n) + N(0, 0.09)
    Y = sin_term + noise_vector

    return X::Vector{Float64},Y::Vector{Float64}
end


