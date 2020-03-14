import numpy as np 
from scipy.optimize import minimize
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import sys


def figure_drives():
	fig, ax = plt.subplots( 1, 2 )
	sigma = 0.6

	x_d = np.linspace( -0.0, 20.0, 100 )
	x_s = np.linspace( 0.0, 1.0, 100 )
	N = lambda x: 1.0/(1 + np.exp(-sigma*x))
	U = lambda x: x**2

	ax[0].plot( x_d, N(x_d - 8.0), linewidth = 3.0 )
	ax[0].plot( [15.0]*2, [0.0, 1.0], 'k--')
	ax[0].set_title('Normalization')
	ax[0].set_xlabel('Physiological quantity')
	ax[0].set_ylabel('Utility')
	ax[1].plot( x_s, U(x_s), linewidth = 3.0 )
	ax[1].set_title('Urgency')
	ax[1].set_xlabel('Utility')
	ax[1].set_ylabel('Drive')

	plt.show() 

def figure_roots():
	a = 0.0
	bs = np.linspace(0.0, 0.1, 100)

	for i in range(len(bs)):
		b = bs[i]
		print b
		U = lambda x: (x**2*(1.0-x)**2 + a*x**2 + b*(1.0-x)**2) 

		xx = b + 2*b**2
		m = minimize( U, 0.0 )

		print "theory: ", xx, ", actual: ", m.x

def figure_motivation_drinking():
	I0 = -0.5
	I1 = 1.5
	a = 0.0
	b = 0.0
	r1 = 0.0
	r2 = 1.0
	sigma = 0.14

	x = np.linspace(I0, I1, 100)
	Q = lambda x: -(r1 - x)*(r2 - x)*((r1 + r2)/2.0 - x)
	L = lambda x: -(a + b)*x/2.0 + (a*r1 + b*r2)/2.0
	U = lambda x: (r1 - x)**2*(r2 -x)**2 + a*(r1 - x)**2 + b*(r2 - x)**2 
	Up = lambda x: Q(x) - L(x)
	
	T = 800.0
	h = 0.01
	N = int(T/h)
	print N
	X = np.zeros(N)
	time = np.zeros(N)
	X[0] = r1

	for i in range(N-1):
		X[i+1] = X[i] - h*Up(X[i]) + np.sqrt(h)*np.random.normal(loc = 0.0, scale=sigma)
		time[i+1] = time[i]+h

	P_inf = lambda x: np.exp(-U(x)/(sigma**2))
	
	fig, ax = plt.subplots(1,2)

	ax[0].plot( x, U(x), 'k', linewidth = 3.0, label = 'Potential' )
	ax[0].plot( x, P_inf(x), 'k--', label = r'$P_\infty$' )
	ax[0].set_title('Motivational potential')
	ax[0].legend( loc = 'upper right')
	ax[0].set_xlabel(r'$\rho$')

	ax[1].plot( time, X )
	ax[1].set_xlabel('time')
	ax[1].set_ylabel(r'$\rho$')
	ax[1].set_title('Motivational dynamics')
	plt.axhline(y = 0, linewidth=1.0, color='k', linestyle = '--')
	plt.axhline(y = 1, linewidth=1.0, color='k', linestyle = '--')
	plt.show()

def get_escape_time( r1, r2, a, b, sigma):
	MAX_TIME = 20000.0

	h = 0.01
	x = r1
	t = 0.0

	# simulate model until escape
	Q = lambda x: -(r1 - x)*(r2 - x)*((r1 + r2)/2.0 - x)
	L = lambda x: -(a + b)*x/2.0 + (a*r1 + b*r2)/2.0
	U = lambda x: (r1 - x)**2*(r2 -x)**2 + a*(r1 - x)**2 + b*(r2 - x)**2 
	Up = lambda x: Q(x) - L(x)

	f = 0

	while True:
		x = x - h*Up(x) + np.sqrt(h)*np.random.normal(loc = 0.0, scale=sigma)
		t += h

		if x > 0.5 and f == 0:
			te = t
			
		if x > 0.7 and f == 0:
			f = 1

		if x < 0.5 and f == 1:
			tb = t - te
			return te, tb

		if t > MAX_TIME:
			raise Exception('Maximal time reached!', 'Max time')


def escape_times():
	a = 0.0
	r1 = 0.0
	r2 = 1.0
	sigma = 0.10
	h = 0.01
	m = 10
	N = 40
	bs = np.linspace(0.0, 0.125, m)
	emu = np.zeros(m)
	edevs = np.zeros(m)
	bmu = np.zeros(m)
	bdevs = np.zeros(m)

	for k in range(m):
		b = bs[k]		
		print "Computing escape time for b = ", b
		etimes = np.zeros(N)
		btimes = np.zeros(N)

		for i in range(N):
			etimes[i], btimes[i] = get_escape_time( r1, r2, a, b, sigma )	

		emu[k] = np.mean(etimes)
		edevs[k] = np.std(etimes)/np.sqrt(N)
		bmu[k] = np.mean(btimes)
		bdevs[k] = np.std(btimes)/np.sqrt(N)

	fig, ax = plt.subplots(1,2)
	ax[0].bar( bs, emu, 0.125/float(m), yerr = edevs )
	ax[0].set_xlabel('Level of thirst')
	ax[0].set_title('Latency to drink')
	ax[0].set_ylabel('Average time')
	ax[1].bar( bs, bmu, 0.125/float(m), yerr = bdevs )
	ax[1].set_xlabel('Level of thirst')
	ax[1].set_ylabel('Average time')
	ax[1].set_title('Bout duration')
	plt.show()
			


if __name__ == '__main__':
    v = sys.argv[1]

    if v == 'drives':
    	figure_drives()
    elif v == 'roots':
    	figure_roots()
    elif v == "drinking":
    	figure_motivation_drinking()
    elif v == "escape":
    	escape_times()
    else:
    	print "Unrecognized option"
    #plot_inhibition_potential()