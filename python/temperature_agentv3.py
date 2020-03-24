import numpy as np 
import matplotlib.pyplot as plt
# Maps
from brains import *
import time

# --------------------------------------------------------------------
# Agent: Represents the physical body of the 
# --------------------------------------------------------------------
class Agent:

	def __init__(self, x, y, theta, brain, Tb = 37.0, E = 0, radius = 2.0 ):
		Tb0 = Tb
		E0 = E
		rho0 = 0.0

		self.brain = brain

		self.s0 = np.array([x, y, theta, Tb0, E0, rho0]); # x, y, theta, Tb, E, rho
		self.A = 2*np.pi*radius
		self.radius = radius
		self.enviroment = None
		self.initSensors( x, y, theta )
		self.F = 0
		self.state = None
		self.orientation = [np.cos(theta), np.sin(theta)]
		self.U = None
		self.trace = None
		self.data = None

	def getTransformation( self, x, y, theta ):
		return np.array([[np.cos(theta), -np.sin(theta), x], 
						 [np.sin(theta), np.cos(theta), y], 
						 [0, 0, 1]])

	def updateSensorPositions( self, x, y, theta ):
		u = np.array([-np.sin(theta), np.cos(theta)])

		signs = { 'T_left': -1,
				'T_right': 1, 
				'F_left': -1,
				'F_right': 1 }

		for key, s in self.sensors.items():
			pos = s
			pos[[0,1]] = signs[key]*self.radius*u + [x,y]
			self.sensors[key] = pos

	def initSensors(self, x, y, theta):
		pl = np.array([self.radius, 0, 1])
		pr = np.array([-self.radius, 0, 1])

		M = self.getTransformation( x, y, theta )

		self.sensors = { 'T_left': np.dot(M,pl), 
					'T_right': np.dot(M,pr), 
					'F_left': np.dot(M,pl),
					'F_right': np.dot(M,pr) }
	def init( self, tf, h ):
		m = int(tf/h)

		self.trace = np.zeros( (2, m) )
		self.state = np.zeros( (self.s0.size, m) )
		self.state[:,0] = self.s0
		self.data = {'mu': np.zeros(m), 'DT': np.zeros(m), 'DF': np.zeros(m), 'a': np.zeros(m), 'b':np.zeros(m)}


	def getSensorData( self, sensor ):
		if self.enviroment is None:
			print('Error: no enviroment set')

		pos = self.sensors[sensor]

		if( sensor in ['T_left', 'T_right'] ):
			return self.enviroment.getTemperature( pos[0], pos[1] )
		else:
			return self.enviroment.getFoodSignal( pos[0], pos[1] )

	# def dmap( self, u, t ):
	# 	Tl = self.getSensorData( 'T_left' )
	# 	Tr = self.getSensorData( 'T_right' )
	# 	Fr = self.getSensorData( 'F_left' )
	# 	Fl = self.getSensorData( 'F_right' )	

	# 	# return temperature_map( u, self.Tp, Tl, Tr, self.G, self.k1, 0.0, self.A )
	# 	# return food_map( u, Fl, Fr, self.G, self.F, 0.9 )
	# 	state_i = drive_map()
	# 	state_motiv = motivation_map()
	# 	state_incentives = incentive_map()
	# 	state_motor = motor_map()

	# 	state, U, mu, DT, DF, a, b = competition_map( u, self.Tp, Tl, Tr, Fr, Fl, self.G, self.k1, 0.0, self.A, self.F, 0.99 )
	# 	self.U = U
	# 	return state, mu, DT, DF, a, b


	def step( self, c_step, h, t ):
		dr, params = self.brain.dmap( self.state[:, c_step], t )
		self.state[:, c_step + 1] = self.state[:, c_step] + h*dr
		# self.state[4] = np.heaviside(self.state[4], 0.5)*self.state[4]
		# self.data['mu'][c_step+1] = mu
		# self.data['DT'][c_step+1] = DT
		# self.data['DF'][c_step+1] = DF
		# self.data['a'][c_step+1] = a
		# self.data['b'][c_step+1] = b
		# moving/ acting
		self.F = 0
		self.theta = np.atan(dr[1]/dr[0])
		self.x = self.state[0, c_step + 1]
		self.y = self.state[1, c_step + 1]
		
		self.updateSensorPositions( self.x, self.y, self.theta )
		self.F = self.enviroment.getFood( self.x, self.y )
		self.trace[:,c_step] = [self.x, self.y]

	def draw( self, c_step, offset ):
		plt.plot( self.trace[0,:c_step], self.trace[1,:c_step], 'k--')
		x = self.x + offset[0]
		y = self.y + offset[1]
		c = plt.Circle( (x, y), self.radius, color = 'k' )
		p1 = self.sensors['T_left'] 
		p2 = self.sensors['T_right'] 
		cs1 = plt.Circle( (p1[0] + offset[0], p1[1] + offset[1]), 1, color = [0.5,0.5,0.5] )
		cs2 = plt.Circle( (p2[0] + offset[0], p2[1] + offset[1]), 1, color = [0.5,0.5,0.5] )
		fig = plt.gcf()
		ax = fig.gca()
		ax.add_artist(c)
		ax.add_artist(cs1)
		ax.add_artist(cs2)
		

# --------------------------------------------------------------------
# Enviroment: Manages enviroment drawing and signals
# --------------------------------------------------------------------
class Enviroment:
	def __init__( self ):
		self.w = 100
		self.h = 100
		self.food_sources = []
		sigma = 30.0
		self.g = lambda x,y,x0,y0: np.exp(-(( x - x0)**2 + (y - y0)**2)/(2*sigma**2) )
		self.Tmin = 15.0
		self.Tmax = 45.0
		self.T = lambda x,y : ((self.Tmax - self.Tmin)/self.w)*x + self.Tmin
		self.time = None
		self.ax_gradient = None

	def addFoodSource( self, x, y ):
		self.food_sources.append((x, y))

	def getTemperature( self, x, y ):
		return self.T(x,y)

	def getFoodSignal( self, x, y ):		
		signal = 0

		for i in range(len(self.food_sources)):
			x0,y0 = self.food_sources[i]
			signal = max( signal, self.g( x,y,x0,y0 ))

		return signal

	def getFood( self, x, y ):		
		# print('Getting food source!') 
		for i in range(len(self.food_sources)):
			x0,y0 = self.food_sources[i]
			# print('Position of the food source: {},{}'.format(x0,y0))

			if( np.linalg.norm(np.array([x-x0, y-y0])) < 3.0 ):
				print( 'Got food source' )
				return 1.0
			
		return 0.0

	def draw( self, offset ):
		delta = 1.0
		cax = plt.gca()

		if self.ax_gradient is None:
			self.ax_gradient = plt.axes([0.07, 0.83, 0.4, 0.09])	

		plt.sca( self.ax_gradient )
		self.ax_gradient.tick_params(bottom = False, left = True, top = False, labelbottom = False, labelleft = True)
		self.ax_gradient.axis([0, self.w, self.Tmin, self.Tmax])
		plt.yticks([self.Tmin, self.Tmax], [self.Tmin, self.Tmax])
		xx = [0, self.w, self.w]
		yy = [self.Tmin, self.Tmin, self.Tmax]
		self.ax_gradient.fill( xx, yy, color=[0.8,0.5,0.5] )

		plt.sca(cax)

		for i in range(len(self.food_sources)):
			x0,y0 = self.food_sources[i]
			x0 += offset[0]
			y0 += offset[1]

			x = np.arange(0, self.w, delta)
			y = np.arange(0, self.h, delta)

			X, Y = np.meshgrid(x, y)
			#Z = np.exp(-(X - x0*np.ones(X.shape))**2 - (Y - y0*np.ones(Y.shape))**2)
			Z = self.g(X,Y, x0*np.ones(X.shape), y0*np.ones(Y.shape))		
			CS = plt.contour(X, Y, Z, origin = 'lower' )

			c = plt.Circle( (x0, y0), 1.0, color = 'g' )
			fig = plt.gcf()
			ax = fig.gca()
			ax.add_artist(c)


# --------------------------------------------------------------------
# Simlation: Manages the simulation
# --------------------------------------------------------------------
class Simulation:
	def __init__(self):
		self.h = 0.01
		self.agents = []
		self.enviroment = Enviroment()
		self.fig2, self.ax2 = plt.subplots(2,2, figsize = (10,5))
		self.fig1, self.ax = plt.subplots(figsize = (10,5) )
		self.ax.set_position([0.07, 0.1, 0.4, 0.7])
		self.ax_temp = plt.axes([0.52, 0.7, 0.46, 0.2])	
		self.ax_energy = plt.axes([0.52, 0.42, 0.46, 0.2])	
		self.ax_potential = plt.axes([0.52, 0.10, 0.46, 0.2])
		self.observed = None
		self.offset = np.array([0,0])

	def addAgent( self, a, observe = 1.0 ):
		a.enviroment = self.enviroment
		self.agents.append( a )
		self.observed = len(self.agents)-1

	def addFoodSource( self, x, y ):
		self.enviroment.addFoodSource( x, y )

	def draw( self, c_step, tf ):
		# plt.subplots_adjust(left=0.1, bottom=0.25, right = 0.9, top = 0.95)
		#plt.ioff()
		pre = time.time()
		a = self.agents[0]
		self.ax2[0,0].cla()
		self.ax2[0,0].set_title('Rho (Motivation)')
		self.ax2[0,0].plot(a.state[5,:])

		self.ax2[0,1].cla()
		self.ax2[0,1].set_title('Drives')
		self.ax2[0,1].plot(a.data['DT'])
		self.ax2[0,1].plot(a.data['DF'])

		self.ax2[1,0].cla()
		self.ax2[1,0].set_title('Mu')
		self.ax2[1,0].plot(a.data['mu'])
		self.ax2[1,0].set_xlabel('time')
		# self.ax2[1,0].ylabel('mu(t)')

		self.ax2[1,1].cla()
		self.ax2[1,1].set_title('Normalized drives')
		self.ax2[1,1].plot(a.data['a'])
		self.ax2[1,1].plot(a.data['b'])
		self.ax2[1,1].set_xlabel('time')

		pos = time.time()

		print('Time to plot variables: ', (pos-pre))

		pre = time.time()
		plt.cla()
		plt.axis([0, self.enviroment.w, 0, self.enviroment.h])
		#plt.title('Motivational conflict motivation')
		plt.xlabel('x')
		plt.ylabel('y')
		# Setting figure properties
		plt.ion()		

		for a in self.agents:
			if a.x + self.offset[0] > self.enviroment.w :
				self.offset[0] = self.enviroment.w + self.offset[0] - a.x - 10.0
			if a.y + self.offset[1] > self.enviroment.h:
				self.offset[1] = self.enviroment.h + self.offset[1] - a.y - 10.0
			if a.x + self.offset[0] < 0:
				self.offset[0] = -a.x + self.offset[0] + 10.0
			if a.y + self.offset[1] < 0:
				self.offset[1] = -a.y + self.offset[1] + 10.0


		self.enviroment.draw( self.offset )

		for a in self.agents:
			a.draw( c_step, self.offset )

		pos = time.time()

		print('Time to plot agents and enviroment: ', (pos-pre))

		pre = time.time()

		a = self.agents[self.observed]	
		plt.sca(self.ax_temp)	
		plt.cla()
		self.ax_temp.plot(self.time[:c_step], a.state[3,:c_step])
		self.ax_temp.axis([0, tf, 0, 45])	
		self.ax_temp.set_title('Tb')	
		plt.sca(self.ax_energy)
		plt.cla()
		self.ax_energy.plot(self.time[:c_step], a.state[4,:c_step])
		self.ax_energy.axis([0, tf, 0, 1.1])
		self.ax_energy.set_title('E')
		plt.sca( self.ax_potential )
		plt.cla()

		if a.U is not None:
			x = np.linspace(-1.0, 2.0, 100)
			self.ax_potential.plot( x, a.U(x))

		self.ax_potential.set_title('Potential')
		plt.sca(self.ax)

		pos = time.time()

		print('Plotting main variables (drives, potential): ', (pos-pre))
		#ax.margins(x = 0)
		plt.pause(0.001)


	def run( self, tf ):
		c_step = 0
		t = 0.0
		m = int(tf/self.h)
		self.time = np.zeros((m, ))

		for a in self.agents:
			a.init( tf, self.h )

		while( c_step < m-1 ):
			self.time[c_step] = t

			pre = time.time()

			for a in self.agents:
				a.step( c_step, self.h, t )

			pos = time.time()

			print('Running a step of the simulation: ', (pos-pre))

			self.draw( c_step, tf )
			c_step += 1
			t += self.h

		plt.ioff()
		plt.show()

if __name__ == '__main__':
	s = Simulation()
	a = Agent( x = 90.0, y = 80.0, theta = np.pi, Tb = 37.0 )
	s.addAgent( a )
	s.addFoodSource( 40, 40  )
	s.run( 20.0 )
