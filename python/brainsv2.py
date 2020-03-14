import numpy as np
from abc import ABC

class Brain (ABC):
	def dmap( self, u, sensors ):
		pass

class SimpleBrain( Brain ):
	def __init_( self ):
		pass

	def dmap( self, u, sensors ):
		Tl = sensors['Tl']
		Tr = sensors['Tr']

		Tb = (Tl + Tr)/2.0
		T = np.array([Tl, Tr])
		alpha = 1.0
		beta = 0.0
		M = np.matrix([[alpha, beta],[beta, alpha]])
		dx = M*T

		params = []

		return dx, params


class MotivationalBrain( Brain ):
	def __init_( self, k1 = 0.99, G = 0.4, Tp = 37.0, Ep = 1.0, A = 1.0 ):
		self.G = G # Heat generation rate
		self.k1 = k1
		self.Tp = Tp
		self.Ep = Ep
		self.A = A

	def drive_map( u, up, Tl, Tr, Fr, Fl, G, k1, k2, A, F )
		Tb = u[0]
		E = u[1]
		Tp = up[0]
		Ep = up[1]

		# Ambient temperature
		Ta = (Tl + Tr)/2.0

		# Food helper functions
		
		I = 10.0
		sigma2 = 0.05
		fe = lambda x : np.pi*x if abs(x)<1.0 else np.pi*np.sign(x)
		# food drives
		wE = 1.0
		mu_food = ge(dFood)
		# Temperature drives
		wH = 1/np.abs(40.0 - 20.0)
		dHeat = (1 + wH*np.abs(Tb - Tp))**2 - 1
		# Motivation
		mu_temperature = gt(dHeat)
		#print( 'mu_food: {}, E: {},  mu_temperature: {}'.format(mu_food, E-Ep, mu_temperature))
		alpha = 0.5
		Tc = Tb # No contact
		dTb = G - k1*(Tb - Ta)*A - k2*(1 - A)*(Tb - Tc)
		dE = -alpha*G + F

		return np.array([Tb, E])

	def motivation_map( u, up, u_drive )
		rho = u[0]
		# Internal state
		Tb = u_drive[0]
		E = u_drives[1]
		Tp = up[0]
		Ep = up[1]

		# Drive normalization
		sigma_t = 15.0
		G_t = 10.0
		x0_t = 0.5
		sigma_e = 10.0
		G_e = 10.0
		x0_e = 1.0
		gt = lambda x : G_t/(1 + np.exp(-sigma_t*(x-x0_t)))
		ge = lambda x : G_e/(1 + np.exp(-sigma_e*(x-x0_e)))
		dHeat = gt(Tp - Tb)
		dFood = ge(Ep - E)
		# Competition parameters
		a = dHeat
		b = dFood

		U = lambda rho: (1.0/4.0)*rho**2*(1 - rho)**2 + a*rho**2 + b*(1 - rho)**2
		dU = lambda rho: (1.0/2.0)*(rho*((1-rho)**2 + a) - (1-rho)*(rho**2 + b))

		noise = np.random.normal(loc = 0.0, scale = 0.1)
		dRho = -10*dU( rho ) + noise

		return np.array([rho])

	def incentives_map():
		TEMP_STATE, FOOD_STATE = 0, 1
		mu_heat = gc(rho, TEMP_STATE)
		mu_food = gc(rho, FOOD_STATE)
		mu = mu_heat + mu_food	



	def motor_map( u, up, u_motivation, u_drive )
		theta = u[3]
		vmax = 5.0
		sigma_c = 0.5
		

		ft = lambda x : np.pi*x/15.0 if abs(x)<15.0 else np.pi*np.sign(x)
		gc = lambda x,x0: np.exp(-(x - x0)**2/(2*sigma_c**2))/np.sqrt(2*np.pi*sigma_c**2)

		dTheta_temperature = ft((Tb - Tp)*(Tl - Tr)) + np.random.rand() - 0.5
		dTheta_energy = fe((Fl - Fr)*(Ep - E)) + np.random.rand() - 0.5

		dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)]) + 5.0*(1/(0.2 + mu))*np.array([np.random.rand(), np.random.rand()])
		dTheta = gc(rho,0)*dTheta_temperature + mu_food*dTheta_energy

		return np.array([dx[0], dx[1], dTheta])

	def dmap( self, u, sensors ):
		pass