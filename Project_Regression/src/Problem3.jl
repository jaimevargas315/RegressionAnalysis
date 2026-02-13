using Statistics
using DelimitedFiles
using Plots
include("Problem1.jl")  # For noisyLinearMultiDim

function bivariate(X::Vector{Float64}, Y::Vector{Float64})
    Xˉ = mean(X)
    Yˉ = mean(Y)
    covariance = mean((X .- Xˉ) .* (Y .- Yˉ))
    variance_X = mean((X .- Xˉ).^2)
    b = covariance / variance_X
    a = Yˉ - b * Xˉ
    return a::Float64, b::Float64
end 