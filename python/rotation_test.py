import numpy as np
import matplotlib.pyplot as plt

F1 = 10.0
F2 = 5.0
radius = 2.0
N = 1000

theta0 = 0.0
dtheta0 = 0.0
v0 = 0.0

k = 1.0
m = 1.0
d = radius
Ic = m*k**2.0
theta = 0.0
hh = 0.01

L = 5.0

theta = theta0
dtheta = dtheta0
v = v0
x = np.array([0.0, 0.0])

plt.ion()
mu = 10.0

for i in range(N):
	print "Iteration: ", i

	if i > 70:
		F1 = 0.0
		F2 = 0.0
	
	Fmu = mu*np.linalg.norm(v) if (np.linalg.norm(v) > 0.0) and (F1 < 0.01 and F2 < 0.01) else 0.0

	F_left = F1 - Fmu
	F_right = F2 - Fmu

	dv = (F_left + F_right)/m
	ddtheta = (d*F_left - d*F_right)/Ic

	# if all(ddtheta) > 0.0:
	# 	r = -dv/ddtheta
	# else:
	# 	r = x

	v = v + hh*dv
	dtheta = dtheta + hh*ddtheta
	theta = theta + hh*dtheta

	print "v; ", v
	print "theta: ", theta

	dx = np.array([v*np.cos(theta), 
		  v*np.sin(theta)])

	print "dx: ", dx

	x = x + hh*dx

	#plt.cla()
	plt.plot( x[0], x[1], 'k.', markersize = 10)
	plt.axis([-L, L, -L, L])
	plt.show()
	plt.pause(0.01)


plt.ioff()


