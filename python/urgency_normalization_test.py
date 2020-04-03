import numpy as np 
import matplotlib.pyplot as plt 

sigma = 0.5
x0 = 0.0
N = lambda x: 2.0/(1 + np.exp(-sigma*(x - x0))) - 1.0
U = lambda x: x**3

x = np.linspace( -5, 5, 100 )

plt.plot( x, U(N(x)) )
#plt.plot( x, N(x) )
plt.show()

