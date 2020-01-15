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
	
	# Parameters
	k2 = 1.0
	Tc = Tb # No contact
	alpha = 0.5

	# Diff Equations	
	dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)])
	dTb = G - k1*(Tb - Ta)*A - k2*(1 - A)*(Tb - Tc)
	dE = -alpha*G
	dTheta = f((Tb - Tp)**3*(Tl - Tr)) + np.random.rand() - 0.5

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
	mu = g(dFood)*(1 - F) + h(E)
	vmax = 5.0
	alpha = 0.5

	# Diff Equations	
	dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)])
	dTb = 0.0
	dE = -alpha*G + F
	
	dTheta = f((Fl - Fr)*(Ep - E)*300) + np.random.rand() - 0.5

	return np.array([dx[0], dx[1], dTheta, dTb, dE ])


def competition_map( u, Tp, Tl, Tr, Fr, Fl, G, k1, k2, A, F, Ep ):
	# State variables	
	theta = u[2]
	Tb = u[3]
	E = u[4]
	rho = u[5]

	# Competition helper function
	sigma_c = 0.5
	gc = lambda x,x0: np.exp(-(x - x0)**2/(2*sigma_c**2))/np.sqrt(2*np.pi*sigma_c**2)

	# Temperature helper functions
	sigma_t = 15.0
	G_t = 10.0
	x0_t = 0.5
	ft = lambda x : np.pi*x/15.0 if abs(x)<15.0 else np.pi*np.sign(x)
	gt = lambda x : G_t/(1 + np.exp(-sigma_t*(x-x0_t)))
	# Ambient temperature
	Ta = (Tl + Tr)/2.0

	# Food helper functions
	sigma_e = 10.0
	G_e = 10.0
	x0_e = 1.0
	I = 10.0
	sigma2 = 0.05
	fe = lambda x : np.pi*x if abs(x)<1.0 else np.pi*np.sign(x)
	ge = lambda x : G_e/(1 + np.exp(-sigma_e*(x-x0_e)))
	he = lambda x : I*np.exp(-(x - Ep)**2/(2*sigma2**2))
	# food drives
	wE = 1.0
	dFood = (1 + wE*(Ep - E))**2 - 1.0
	mu_food = dFood#ge(dFood)
	# Temperature drives
	wH = 1/np.abs(40.0 - 20.0)
	dHeat = (1 + wH*np.abs(Tb - Tp))**2 - 1
	# Motivation
	mu_temperature = dHeat#gt(dHeat)
	print( 'mu_food: {}, E: {},  mu_temperature: {}'.format(mu_food, E-Ep, mu_temperature))
	mu = (mu_food + mu_temperature)*(1 - F) + he(E)
	vmax = 5.0
	alpha = 0.5
	Tc = Tb # No contact
	# Competition parameters
	a = np.abs(dHeat)
	b = np.abs(dFood)
	n1 = (np.abs(a) + np.abs(b))
	a = a/n1
	b = b/n1
	#print( 'a: {}, b: {}, rho: {}'.format(a, b, rho) )	
	# Potential
	U = lambda rho: (1.0/4.0)*rho**2*(1 - rho)**2 + a*rho**2 + b*(1 - rho)**2
	dU = lambda rho: (1.0/2.0)*(rho*((1-rho)**2 + a) - (1-rho)*(rho**2 + b))

	#print((1/(0.2 + mu)))
	dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)]) + 5.0*(1/(0.2 + mu))*np.array([np.random.rand(), np.random.rand()])
	dTb = G - k1*(Tb - Ta)*A - k2*(1 - A)*(Tb - Tc)
	dE = -alpha*G + F
	dTheta_temperature = ft((Tb - Tp)**3*(Tl - Tr)) + np.random.rand() - 0.5
	dTheta_energy = fe((Fl - Fr)*(Ep - E)*300) + np.random.rand() - 0.5
	dTheta = gc(rho,0)*dTheta_temperature + gc(rho, 1)*dTheta_energy
	dRho = -10*dU( rho )

	return np.array([dx[0], dx[1], dTheta, dTb, dE, dRho ]), U
