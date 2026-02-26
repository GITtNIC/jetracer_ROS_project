from math import *
import numpy as np
import matplotlib.pyplot as plt

L = 0.15
R = np.array([-0.81,-0.96,-1.05,-1.13,-1.28,-1.45,-1.57,-2.12, -2.64, -5.198, 9999999.0, 5.99, 2.55, 1.88, 1.68, 1.37, 1.14, 0.99, 0.81, 0.755, 0.75])
x = np.arctan(2*L/R)*180/np.pi
x1 = x/180*np.pi
y = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
y1 =  1500 - y/np.pi*2000
print(x)
print(x1)
print(y1)
p = np.polyfit(x, y1, 3) 

#np.savetxt('output.txt', p,fmt='%f',delimiter=" ")


print(p)
x_test = []
y_test = []
for i in range(-25, 25):
    x_test.append(i)
    y_test.append(np.polyval(p, i))
    
plt.plot(x_test, y_test, 'r')
plt.plot(x, y1, 'g')
plt.show()

