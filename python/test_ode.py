import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

fig = plt.figure(1, figsize=(5,5))

def f(y, t):
	return y**2

y0 = 1
t = np.linspace(0, 0.9, 100)
ysol = odeint( f, y0, t )

ax1 = fig.add_subplot( 121 )
ax1.plot( t, ysol )
c = -1/y0

ax2 = fig.add_subplot( 122 )
ax2.plot( t, -1.0/(t + c))
plt.show()

