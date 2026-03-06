
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# Question 1 — CDF Probabilities
print("----- Question 1 -----")
P_X_greater_5 = math.exp(-5)
P_X_less_5 = 1 - math.exp(-5)
P_between_3_7 = math.exp(-3) - math.exp(-7)

# Monte Carlo Simulation
N = 100000
samples = np.random.exponential(scale=1, size=N)
sim_P_X_greater_5 = np.mean(samples > 5)
print("Analytical P(X > 5):", P_X_greater_5)
print("Simulated  P(X > 5):", sim_P_X_greater_5)
print("Analytical P(X < 5):", P_X_less_5)
print("Analytical P(3 < X < 7):", P_between_3_7)

# Question 2 — PDF Validation and Plot

print("\n----- Question 2 -----")

def pdf(x):
    return 2*x*np.exp(-x**2)
    
x_vals = np.linspace(0,10,100000)
# Non-negativity
non_negative = np.all(pdf(x_vals) >= 0)
# Numerical Integration
integral = np.trapezoid(pdf(x_vals), x_vals)
# Valid PDF
is_valid_pdf = abs(integral - 1) < 1e-3
print("Integral value:", integral)
print("Non-negative:", non_negative)
print("Valid PDF:", is_valid_pdf)
# Plot
x_plot = np.linspace(0,3,400)
plt.plot(x_plot, pdf(x_plot))
plt.title("PDF: f(x) = 2x e^(-x^2)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# Question 3 — Exponential Distribution
print("\n----- Question 3 -----")
# Analytical
P_X_greater_5 = math.exp(-5)
P_between_1_3 = math.exp(-1) - math.exp(-3)
# Monte Carlo
samples = np.random.exponential(scale=1, size=N)

sim_P_X_greater_5 = np.mean(samples > 5)
sim_P_between = np.mean((samples > 1) & (samples < 3))

print("Analytical P(X > 5):", P_X_greater_5)
print("Simulated  P(X > 5):", sim_P_X_greater_5)

print("Analytical P(1 < X < 3):", P_between_1_3)
print("Simulated  P(1 < X < 3):", sim_P_between)

# Question 4 — Gaussian Distribution
print("\n----- Question 4 -----")
mu = 10
sigma = 2
# Analytical
P_X_le_12 = norm.cdf(12, mu, sigma)
P_between = norm.cdf(12, mu, sigma) - norm.cdf(8, mu, sigma)
# Monte Carlo
samples = np.random.normal(mu, sigma, N)
sim_P_X_le_12 = np.mean(samples <= 12)
sim_P_between = np.mean((samples > 8) & (samples < 12))

print("Analytical P(X ≤ 12):", P_X_le_12)
print("Simulated  P(X ≤ 12):", sim_P_X_le_12)

print("Analytical P(8 < X < 12):", P_between)
print("Simulated  P(8 < X < 12):", sim_P_between)
