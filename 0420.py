import numpy as np
import matplotlib.pylab as plt

x1 = np.array([1, -1, 800, -5, 1000])


def step_function(x):
    return np.array(x > 0, dtype=np.int8)


print(step_function(1))
print(step_function(x1))

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

print(y)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
