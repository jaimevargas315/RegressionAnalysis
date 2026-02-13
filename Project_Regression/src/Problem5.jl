using LinearAlgebra: pinv
using DelimitedFiles
using Statistics
using Plots
include("Problem2.jl")  # For noisySin
function polyfitLS(𝐗::Vector{Float64}, 𝐘::Vector{Float64}, M::Integer)

    # 1. Define dimensions and perform initial check
    N = length(𝐗)

    # The number of coefficients is M + 1 (for powers X^0 through X^M)
    if N < M + 1
        error("M must be less than the length of 𝐗 minus one (N ≥ M + 1)")
    end

    # 2. Construct the Design Matrix (Φ) - N x (M+1)
    𝚽 = zeros(N, M + 1)

    # Populate the columns of Φ with powers of X
    for j in 0:M
        𝚽[:, j+1] = 𝐗 .^ j
    end

    # 3. Calculate the Weight Vector using the Normal Equation - w = (ΦᵀΦ)⁻¹ * ΦᵀY

    # Calculate ΦᵀΦ 
    𝚽T𝚽 = 𝚽' * 𝚽

    # Calculate ΦᵀY 
    𝚽TY = 𝚽' * 𝐘

    #  Solve (ΦᵀΦ * w = ΦᵀY) for w.
    # Julia's left-division operator (\) is used.
    𝐰 = 𝚽T𝚽 \ 𝚽TY

    return 𝐰::Vector{Float64}
end

function predict_curve(X::Vector{Float64}, w::Vector{Float64})
    M = length(w) - 1 # Polynomial degree
    N = length(X)

    #Construct the Design Matrix (Φ) for prediction data
    Ŷ = zeros(N, M + 1)

    for j in 0:M
        Ŷ[:, j+1] = X .^ j
    end

    # Prediction is the matrix product: Ŷ * w
    return Ŷ * w
end

function calculate_erms(Y_actual::Vector{Float64}, Y_predicted::Vector{Float64})
    # Calculate the squared errors (Y_actual - Y_predicted)^2
    squared_errors = (Y_actual .- Y_predicted) .^ 2

    # Calculate the Mean of the Squared Errors (MSE)
    mse = mean(squared_errors)

    # Take the square root to get E_RMS
    return sqrt(mse)
end