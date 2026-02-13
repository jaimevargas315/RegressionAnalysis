using Statistics
using Plots
using DelimitedFiles
include("Problem1.jl")  # For noisyLinearMultiDim
function multivariate(𝐗₁::Vector{Float64}, 𝐗₂::Vector{Float64}, 𝐘::Vector{Float64})
    m = length(𝐘)
# 1. Construct the Design Matrix (X)
    # The first column is a column of ones for the intercept (a).
    # hcat() horizontally concatenates the columns.
    X = hcat(ones(m), 𝐗₁, 𝐗₂)

    # 2. Calculate the Coefficients using the Normal Equation: B = (X' * X) \ (X' * Y)
    # Normal Equation: B=(X'X)^-1 * X'Y
    # Julia's backslash operator (\) solves the system of linear equations.
    B = (X' * X) \ (X' * 𝐘)

    # 3. Extract and Return the Float64 Coefficients
    # B is a 3-element vector: [a, b1, b2]
    a  = B[1]
    b₁ = B[2]
    b₂ = B[3]
    return a::Float64, b₁::Float64, b₂::Float64
end 