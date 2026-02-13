using LinearAlgebra
function polyfitRegLS(𝐗::Vector{Float64}, 𝐘::Vector{Float64}, M::Integer, λ::Float64)
    # 1. Define dimensions and perform initial check
    N = length(𝐗)
    
    # Check if there are enough data points for the polynomial degree (M+1 coefficients)
    if N < M + 1
        error("N must be greater than or equal to M + 1 to solve the system.")
    end
    
    # 2. Construct the Design Matrix (Φ) - N x (M+1)
    # M+1 columns for powers X^0 (intercept) through X^M
    𝚽 = zeros(N, M + 1) 
    
    # Populate the columns of Φ with powers of X
    for j in 0:M
        # X .^ j computes the j-th power element-wise for all elements in X
        𝚽[:, j + 1] = 𝐗 .^ j 
    end

    # 3. Calculate the components of the Regularized Normal Equation
    
    # Calculate ΦᵀΦ (The relationship between input features)
    𝚽T𝚽 = 𝚽' * 𝚽

    # Calculate ΦᵀY (The relationship between features and target)
    𝚽TY = 𝚽' * 𝐘

    # 4. Apply Regularization (λI)
    
    # I is the Identity Matrix of size (M+1)x(M+1)
    # The term (λ * I) is added to the diagonal of ΦᵀΦ
    # This is the core of Ridge Regression: w* = (λI + ΦᵀΦ)⁻¹ * ΦᵀY
    regularized_matrix = (λ * I) + 𝚽T𝚽

    # 5. Solve for the Weight Vector (w*)
    
    # Julia's left-division operator (\) solves the linear system: 
    # (regularized_matrix * w) = ΦTY for w.
    𝐰 = regularized_matrix \ 𝚽TY

    return 𝐰::Vector{Float64}
end
