import numpy as np

def temperature_map( u, Tp, Tl, Tr, G, k1, k2, A ):

	# Nonlinearities
	sigma = 5.0
	vmax = 5.0
	x0 = 1.0
	f = lambda x : np.pi*x/15.0 if abs(x)<15.0 else np.pi*np.sign(x)
	g = lambda x : vmax/(1 + np.exp(-sigma*(x-x0)))
	# Ambient temperature
	Ta = (Tl + Tr)/2.0
	# State variables
	theta = u[2]
	Tb = u[3]
	E = u[4]
	# Drives
	wH = 1/np.abs(40.0 - 30.0)
	dHeat = (1 + wH*np.abs(Tb - Tp))**2 - 1
	# Motivation
	mu = g(dHeat)
	print( 'mu: {}'.format(mu) )
	
	# Parameters
	k2 = 1.0
	Tc = Tb # No contact
	alpha = 0.5

	# Diff Equations	
	dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)])
	dTb = G - k1*(Tb - Ta)*A - k2*(1 - A)*(Tb - Tc)
	dE = -alpha*G
	dTheta = f((Tb - Tp)**3*(Tl - Tr)) + np.random.rand() - 0.5#np.heaviside(-np.sign(dTb)*np.sign(Tb-self.Tp), 0.0)*0.1

	return np.array([dx[0], dx[1], dTheta, dTb, dE ])


def food_map( u, Fl, Fr, G, F, Ep ):
	sigma = 15.0
	vmax = 10.0
	x0 = 0.5
	I = 10.0
	sigma2 = 0.05
	f = lambda x : np.pi*x if abs(x)<1.0 else np.pi*np.sign(x)
	g = lambda x : vmax/(1 + np.exp(-sigma*(x-x0)))
	h = lambda x : I*np.exp(-(x - Ep)**2/(2*sigma2**2))
	# State variables
	theta = u[2]
	Tb = u[3]
	E = u[4]
	# Drives
	wE = 1.0
	dFood = (1 + wE*(Ep - E))**2 - 1.0
	# Motivation
	print('Satiety: {}'.format(h(E)) )
	mu = g(dFood)*(1 - F) + h(E)
	vmax = 5.0
	alpha = 0.5

	# Diff Equations	
	dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)])
	dTb = 0.0
	dE = -alpha*G + F
	
	dTheta = f((Fl - Fr)*(Ep - E)*300) + np.random.rand() - 0.5

	print( 'Arg: {}, f(Arg): {}'.format((Fl - Fr)*(1 - E)*100, f((Fl - Fr)*(1 - E)*100)) )
	print( 'mu: {}, dTheta: {}'.format(mu, dTheta) )

	return np.array([dx[0], dx[1], dTheta, dTb, dE ])


def competition_map( u ):
	return []
