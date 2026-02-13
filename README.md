This project provides a comprehensive exploration of linear and polynomial regression techniques using Julia. Below is a structured summary of the simulation results, key observations, and the underlying mathematical models implemented in this repository.
## Problem 1 & 2: Data Simulation

The foundation of these experiments lies in generating synthetic data where the "ground truth" is known. This allows for a rigorous evaluation of how well different models recover original parameters.
### Multi-Dimensional Linear Data

In Problem 1, we generated data following the model:
Y=a+i=1∑m​bi​Xi​+ϵ


where ϵ∼N(0,σ2).

    1D Visualization: Visualized a single feature space where the true relationship sits within the noisy data points.

    3D Visualizations: * Dataset A (b=[1.0,0.0]): Shows a plane sloping only along the X1​ axis, demonstrating zero influence from X2​.

        Dataset B (b=[1.0,2.0]): Shows a plane sloping along both axes.

### Sinusoidal Data

In Problem 2, we moved to non-linear data generation using:
Y=sin(2πX)+ϵ


This dataset serves as the benchmark for the polynomial curve fitting and regularization tasks.
## Problem 3 & 4: Regression Analysis

We implemented the Normal Equation to solve for best-fit parameters across different dimensions.
### Bivariate and Multivariate Results

    Bivariate (1D): The model estimated the intercept a and slope b from provided datasets using mean-centered covariance.

    Multivariate (2D): Using a design matrix approach, we solved for multiple coefficients simultaneously.

        The Normal Equation: B=(XTX)−1XTY

        Visualization: The resulting fit is a regression surface (plane) that minimizes the vertical distance to the noisy data points.

## Problem 5: Polynomial Curve Fitting & Overfitting

This section demonstrates the critical trade-off between model complexity (degree M) and generalization.
### The Impact of Model Degree (M)

    M=0,1: High bias (Underfitting). The model is too simple to capture the sine wave.

    M=3: The "Sweet Spot." The curve follows the sine wave closely.

    M=9: High variance (Overfitting). With small datasets (N=10), the model passes through every data point but oscillates wildly between them.

### Root Mean Square Error (ERMS​)

As M increases, the Training Error consistently drops toward zero. However, the Test Error decreases initially but then skyrockets at M=9, a classic signature of the model losing its ability to generalize.
## Problem 6: Regularization (Ridge Regression)

To combat the overfitting seen in higher-degree polynomials, we implemented Ridge Regression (L2 Regularization).
### The Regularized Normal Equation

The weights are calculated by adding a penalty term λ to the diagonal of the matrix:
w∗=(λI+ΦTΦ)−1ΦTY

    Small λ (e.g., e−18): The penalty is too small; the curve remains erratic and overfit.

    Optimal λ (e.g., 1.0): The penalty effectively "shrinks" the coefficients, smoothing the curve and providing a stable approximation of the true sine function.

## Key Takeaways

    Data Volume vs. Complexity: Increasing the number of observations (N) allows complex models to generalize better without needing heavy regularization.

    Regularization is a Safeguard: When data is scarce, λ prevents the model from "memorizing" noise.

    Visual Diagnosis: Comparing the "True Function" vs. the "Estimated Function" provides the most intuitive way to diagnose model health (bias vs. variance).
