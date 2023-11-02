import numpy as np
import math
import matplotlib.pyplot as plt

def func(x):
    return (x - 4) ** 2 + math.log(x)

def grad(x):
    return 2 * (x - 4) + 1 / x

def hess(x):
    return 2 - 1 / (x ** 2)

def newton_method(func, grad, hess, x_0, tolerance=1e-5, max_iter=100):
    x_k = x_0

    for k in range(max_iter):
        f_k = func(x_k)
        plt.scatter(x_k, f_k, color='red', s=40)

        g_k = grad(x_k)
        h_k = hess(x_k)
        print('iteration ', k + 1)
        print('function ', f_k)
        print('point ', x_k)
        print('first d ', g_k)
        print('second d ', h_k)

        if g_k == 0:
            print(k + 1, 'iteration count')
            return x_k  # Stop execution


        d_k = -g_k / h_k  # Newton's step
        print('newton step ', d_k)
        x_k += d_k

        # Check stopping criterion
        if math.fabs(grad(x_k)) < tolerance:
            print(k + 1, 'iteration count')
            return x_k

    return x_k

# Set initial guess
x_initial = 4.8

# Run Newton's method
x_min = newton_method(func, grad, hess, x_initial)
print("Minimum point: ", x_min)
print("Minimum value: ", func(x_min))

xx = np.linspace(3.0, 5.0, 100)
yy = []

for i in range(len(xx)):
    yy.append(func(xx[i]))


plt.plot(xx, yy)
plt.show()

