using Plots
function noisyLinearMultiDim(N::Integer, a::Float64 , b::Vector{Float64}; σ²=0.001)
    # The number of features (m) is determined by the length of the coefficient vector b
    m = length(b)
    
    # 1. Generate the Feature Matrix (X)
    # X_matrix is N rows (observations) by m columns (features).
    # We use rand() to generate features from a standard Uniform distribution (0 to 1).
    X = rand(N, m)
    
    # 2. Compute the deterministic part of the prediction (Intercept + Linear Predictor)
    # The linear term is X_matrix * b.
    linear_predictor = X* b
    
    # Add the intercept 'a' to all predictions
    ŷ = linear_predictor .+ a

    # 3. Generate the vector of random noise terms using built-in randn()
    # randn(N) generates N samples from N(0, 1).
    # Scale it by the standard deviation (sqrt(σ²)) to get N(0, σ²).
    noise_std = sqrt(σ²)
    noise_vector = randn(N) .* noise_std

    # 4. Compute the final noisy prediction vector
    Ŷ = ŷ + noise_vector

    # 5. Return the matrix and the prediction vector as a tuple

    return X::Matrix{Float64},Ŷ::Vector{Float64}
end
