import numpy as np
from recorder import *

class Brain (object):
	def __init__( self ):
		self.sensors = {}
		self.recorder = None
		self.state = None
		self.s0 = None
		self.state_labels = None
		self.dx = [0, 0]
		self.body = None

	def setRecorder( self, recorder ):
		self.recorder = recorder

	def setBody( self, body ):
		self.body = body

	def registerSensor( self, name, sensor ):
		self.sensors[name] = sensor

	def dmap( self, t, h, rewards ):
		pass

	def step(self, c_step, h, t, reward ):

		assert self.body is not None, 'No body model'  

		self.recorder.recordState( self.state, c_step )
		wheel_force, di = self.dmap( t, h, reward )
		
		if self.state is not None and len(di) > 0:
			self.state = self.state + h*di

		self.body.step( c_step, h, wheel_force )

	def reset( self, m ):
		self.state = self.s0

		if self.recorder is not None:
			self.recorder.reset( m, self.s0, [])

class SimpleBrain( Brain ):
	def __init__( self ):
		super(SimpleBrain, self).__init__()
		self.s0 = np.array([]); # x, y, theta, Tb, E, rho
		self.state_labels = []

	def dmap( self, t, h, rewards ):
		Tl = self.sensors['Tl'].getReading()
		Tr = self.sensors['Tr'].getReading()
		
		Tb = (Tl + Tr)/2.0
		e = (Tb - 37.0)/40.0

		print "e : ", e
		if np.abs(Tl - Tr) < 0.1:
			print "Correcting"
			r = np.random.random()
			Tl += np.abs(e) if r > 0.5 else 0.0
			Tr += np.abs(e) if r < 0.5 else 0.0

		
		T = np.array([Tl, Tr])
		
		alpha = np.abs(e) if e < 0 else 0.0
		beta = np.abs(e) if e > 0 else 0.0

		M = np.matrix([[alpha, beta],[beta, alpha]])
		wheel_force = np.dot(M,T)/40.0

		return wheel_force, []

class MotivationalBrain( Brain ):

	def __init__( self, Tb, E, k1 = 0.99, G = 0.4, Tp = 37.0, Ep = 1.0, A = 1.0 ):
		super(MotivationalBrain, self).__init__()
		self.s0 = np.array([Tb, E, 0.0]); # Tb, E, rho
		self.state_labels = ['Tb', 'E', 'Rho']

		self.G = G # Heat generation rate
		self.k1 = k1
		self.Tp = Tp
		self.Ep = Ep
		self.A = A
		self.k2 = 1.0
		self.diff_heat = 0.1

		self.initMaps()

	def initMaps( self ):
		# Drive map temperature
		G_t = 2.0
		sigma_t = 0.9
		T0 = 0.0
		N_temp = lambda T: G_t/(1.0 + np.exp(-sigma_t*(T - T0))) - G_t/2.0
		U_temp = lambda T: T**3
		self.D_temp = lambda T: U_temp( N_temp(T) )
		# Drive map Food
		G_f = 2.0
		sigma_f = 10.0
		x0 = 0.0
		N_food = lambda x: G_f/(1 + np.exp(-sigma_f*(x - x0))) - G_f/2.0
		U_food = lambda x: x**3
		self.D_food = lambda x: U_food( N_food(x) )
		# Incentive maps
		sigma_c = 0.5
		self.xi = lambda x,x0: np.exp(-(x - x0)**2/(2*sigma_c**2))/np.sqrt(2*np.pi*sigma_c**2)

	def drive_map( self, Tb, E, Tl, Tr, Fr, Fl, F ):
		
		# Ambient temperature
		Ta = (Tl + Tr)/2.0

		# Computing physiological state
		alpha = 0.1
		Tc = Tb # No contact
		dTb = self.G - self.k1*(Tb - Ta)*self.A - self.k2*(1 - self.A)*(Tb - Tc)
		dE = -alpha*self.G + F

		# Setting up the drives
		dFood = self.D_food( E - self.Ep )
		dTemp = self.D_temp( Tb - self.Tp )

		print "Drive temp: ", dTemp, "- diff: ", (Tb - self.Tp)
		print "Drive E", dFood
		
		return dTb, dE, dTemp, dFood

	def motivation_map( self, h, rho, dTemp, dFood ):
		L = 10.0

		# Competition parameters
		a = np.abs(dTemp)
		b = np.abs(dFood)

		U = lambda rho: (1.0/4.0)*rho**2*(1 - rho)**2 + a*rho**2 + b*(1 - rho)**2
		dU = lambda rho: (1.0/2.0)*(rho*((1-rho)**2 + a) - (1-rho)*(rho**2 + b))

		noise = np.random.normal(loc = 0.0, scale=self.diff_heat)/np.sqrt(h)
		dRho = -L*dU( rho ) + noise

		return dRho

	def incentives_map( self, rho ):
		TEMP_STATE, FOOD_STATE = 0, 1

		mu_heat = self.xi(rho, TEMP_STATE)
		mu_food = self.xi(rho, FOOD_STATE)

		print "Incentive temp: ", mu_heat
		print "Incentive food: ", mu_food
		
		return mu_heat, mu_food

	# def motor_map( self, u, up, u_motivation, u_drive ):
	# 	vmax = 5.0
	# 	sigma_c = 0.5
		

	# 	ft = lambda x : np.pi*x/15.0 if abs(x)<15.0 else np.pi*np.sign(x)
	# 	gc = lambda x,x0: np.exp(-(x - x0)**2/(2*sigma_c**2))/np.sqrt(2*np.pi*sigma_c**2)

	# 	dTheta_temperature = ft((Tb - Tp)*(Tl - Tr)) + np.random.rand() - 0.5
	# 	dTheta_energy = fe((Fl - Fr)*(Ep - E)) + np.random.rand() - 0.5

	# 	dx = mu*vmax*np.array([np.cos(theta), np.sin(theta)]) + 5.0*(1/(0.2 + mu))*np.array([np.random.rand(), np.random.rand()])
	# 	dTheta = gc(rho,0)*dTheta_temperature + mu_food*dTheta_energy

	# 	return np.array([dx[0], dx[1], dTheta])

	def motor_map( self, mu_temp, mu_food, Dtemp, Dfood, Tl, Tr, Fl, Fr ):
		A = np.matrix([[1.0, 0.0],[0.0, 1.0]])
		B = np.matrix([[0.0, 1.0],[1.0, 0.0]])
		sensors_temp = np.array([Tl, Tr])/30.0
		print sensors_temp
		sensors_food = np.array([Fl, Fr])

		Ta = (Tl + Tr)/2.0
		dT = Ta - self.Tp

		T_app = np.heaviside( dT, 0.5 )*np.abs(Dtemp)
		T_avoid = np.heaviside( -dT, 0.5 )*np.abs(Dtemp)
		F_app = np.heaviside( Dfood, 0.5 )*np.abs(Dfood)
		F_avoid = np.heaviside( -Dfood, 0.5 )*np.abs(Dfood)

		print "T_app: ", T_app, ", T_avoid: ", T_avoid
		F_temp = T_app*B + T_avoid*A
		F_food = F_app*B + F_avoid*A
		M_temp = mu_temp*F_temp
		M_food = mu_food*F_food

		wheel_drive = np.dot(M_temp, sensors_temp) +\
					  np.dot(M_food, sensors_food) + \
					  (np.random.random(2)-0.5)*0.1

		print "Wheel drive: ", wheel_drive

		return wheel_drive

	def dmap( self, t, h, rewards ):
		Tl = self.sensors['Tl'].getReading()
		Tr = self.sensors['Tr'].getReading()
		Fl = self.sensors['Fl'].getReading()
		Fr = self.sensors['Fr'].getReading()

		Tb = self.state[0]
		E = self.state[1]
		rho = self.state[2]

		dTb, dE, dTemp, dFood = self.drive_map(Tb, E, Tl, Tr, Fr, Fl, rewards)
		dRho = self.motivation_map( h, rho, dTemp, dFood )
		mu_temp, mu_food = self.incentives_map( rho )
		# Define wheel forces
		wheel_drive = self.motor_map( mu_temp, mu_food, dTemp, dFood, Tl, Tr, Fl, Fr )

		return wheel_drive, np.array([dTb, dE, dRho])