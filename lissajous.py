from numpy import sin,cos,pi,linspace
from pylab import plot,show,subplot
import numpy as np


# Sines Curves to Generate Lissajous Curves
delta = -1
P = 25
t = np.arange(0,1000,1)

y1 = -0.4 * cos(2 * pi * t / P + delta)
y2 = 0.8 * sin(2 * pi * t / P)
y3 = -0.4 * cos(4 * pi * t / P + delta)
y4 = 0.8 * sin(4 * pi * t / P)
y = np.vstack((y1, y2, y3, y4))


# Gaussian Noise
gauss_n = np.zeros((12,1000));
for n in range (0,3):
    gauss_n[4 * n] = np.random.normal(0, 0.01, 1000)
    gauss_n[4 * n + 1] = np.random.normal(0, 0.03, 1000)
    gauss_n[4 * n + 2] = np.random.normal(0, 0.05, 1000)
    gauss_n[4 * n + 3] = np.random.normal(0, 0.07, 1000)


# Plotting Lissajous Curves
k = 1
for i in range(0, 4):
 for j in range(0, 4):
  if i != j:
   subplot(3, 4, k)
   plot(y[i] + gauss_n[k-1], y[j] + gauss_n[k-1])
   k = k + 1

show()
