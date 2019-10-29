import numpy as np 
import matplotlib.pyplot as plt

c = 0.3
d = 0.1
e = 0.0
U = lambda x: (1.0/4.0)*((x**2)*(1-x)**2*(2-x)**2 + (1-x)**2*c + x**2*d + (2-x)**2*e)
x = np.linspace(-0.2,2.2,100)

plt.plot( x, [U(s) for s in x] )
plt.show()