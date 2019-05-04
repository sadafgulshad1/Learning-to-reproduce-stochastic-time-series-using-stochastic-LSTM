import numpy as np
import matplotlib.pylab as plt
P=25
y1=np.zeros(100)
y2=np.zeros(100)
y3=np.zeros(100)
y4=np.zeros(100)
l1=np.zeros(100)
n=0
for t in range(100):
    stan_dev = [0.01, 0.03, 0.05, 0.07]
    print stan_dev
    y1[t]=-0.4*(np.cos((2*np.pi*t)/P)-1)+np.random.normal(0,stan_dev[0])
    y2[t] = 0.8 * (np.sin((2 * np.pi * t) / P) ) + np.random.normal(0, stan_dev[1])
    y3[t] = -0.4*(np.cos((4*np.pi*t)/P)-1)+np.random.normal(0,stan_dev[2])
    y4[t] = 0.8 * (np.sin((4 * np.pi * t) / P)) + np.random.normal(0, stan_dev[3])
    l1[t]=y1[t]+y2[t]

print y1.size
print y1
t1=np.linspace(1,100,100)
print t1.size
plt.figure(1)
plt.plot(t1,y1)
plt.figure(2)
plt.plot(t1,y2)
plt.figure(3)
plt.plot(t1,y3)
plt.figure(4)
plt.plot(t1,y4)
plt.figure(5)
plt.plot(t1,l1)
plt.show()

