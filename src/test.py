import numpy as np
import matplotlib.pyplot as plt

# Sample points (x, y) - replace these with your actual data
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])  # Perfect quadratic: y = x^2

# Step 1: Fit quadratic y = ax^2 + bx + c
coeffs = np.polyfit(x, y, 2)  # [a, b, c]
a, b, c = coeffs
print(f"Fitted coefficients: a={a:.3f}, b={b:.3f}, c={c:.3f}")

# Step 2: Compute curvature at each point
# y'(x) = 2a*x + b, y''(x) = 2a
y_prime = lambda x: 2 * a * x + b
y_double_prime = 2 * a

def curvature(x_val):
    dy_dx = y_prime(x_val)
    return abs(y_double_prime) / ((1 + dy_dx**2) ** 1.5)

curvatures = [curvature(xi) for xi in x]

# Print curvature at each point
for xi, k in zip(x, curvatures):
    print(f"Curvature at x={xi}: {k:.5f}")


# Optional: Plot curve and curvature
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit**2 + b * x_fit + c
plt.plot(x, y, 'ro', label='Data Points')
plt.plot(x_fit, y_fit, 'b-', label='Fitted Curve')
plt.title('Quadratic Fit and Curvature')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
