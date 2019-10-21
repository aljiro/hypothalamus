import numpy as np 
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Enviroment: Contains the main behaviour of the agent
# --------------------------------------------------------------------
class Agent:

	def __init__(self, x, y, theta, Tb, Tp = 37.0, radius = 2.0, k1 = 0.99, G = 0.4 ):
		self.s0 = np.array([x, y, theta, Tb, 1.0]);
		self.Tp = Tp # Prefered temperature
		self.G = G # Heat generation rate
		self.k1 = k1
		self.A = 2*np.pi*radius
		self.radius = radius
		self.enviroment = None
		self.initSensors( x, y, theta )
		self.F = 0
		self.state = None
		self.orientation = [np.cos(theta), np.sin(theta)]

	def getTransformation( self, x, y, theta ):
		return np.array([[np.cos(theta), -np.sin(theta), x], 
						 [np.sin(theta), np.cos(theta), y], 
						 [0, 0, 1]])

	def updateSensorPositions( self, h, mu, x, y, theta0, theta ):
		alpha = np.dot(self.orientation, theta)
		M = self.getTransformation(0, 0, theta)
		c = np.array([x, y, 0])

		for s in self.sensors:
			pos = self.sensors[s]
			pos[[0,1]] += h*mu*np.array([np.cos(theta0), np.sin(theta0)])
			self.sensors[s] = np.dot(M, pos - c) + c

		self.orientation = M[[0,1],[0,1]]*self.orientation

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

		self.state = np.zeros( (self.s0.size, m) )
		self.state[:,0] = self.s0


	def getSensorData( self, sensor ):
		if self.enviroment is None:
			print('Error: no enviroment set')

		pos = self.sensors[sensor]

		if( sensor in ['T_left', 'T_right'] ):
			return self.enviroment.getTemperature( pos[0], pos[1] )
		else:
			return self.enviroment.getFoodSignal( pos[0], pos[1] )

	def dmap( self, u, t ):
		Tl = self.getSensorData( 'T_left' )
		Tr = self.getSensorData( 'T_right' )
		Fr = self.getSensorData( 'F_left' )
		Fl = self.getSensorData( 'F_right' )

		Ta = (Tl + Tr)/2.0

		print('la: {}, Tr: {}'.format(Tl, Tr))
		# State variables
		theta = u[2]
		Tb = u[3]
		E = u[4]
		# Drives
		dHeat = np.abs(Tb - self.Tp)/np.abs(40.0 - 35.0)
		dFood = np.heaviside(1 - E, 0.0)*(1 - E)
		# Parameters
		mu = 2*np.linalg.norm(np.array([dHeat, dFood]))
		k2 = 1.0
		Tc = Tb # No contact
		alpha = 1.0
		# nonlinearity
		sigma = 0.1
		f = lambda x : 1.0/(1 + np.exp(-sigma*x))
		# Diff Equations	
		dx = mu*np.array([np.cos(theta), np.sin(theta)])
		dTb = self.G - self.k1*(Tb - Ta)*self.A - k2*(1 - self.A)*(Tb - Tc)
		dE = -alpha*self.G + self.F
		
		if( dHeat > dFood ):
			print('Temperature drive: {}'.format(dHeat))
			dTheta = f((Tb - self.Tp)*(Tr - Tl))
		else:
			print('Energy Drives')
			dTheta = f( (E - 0.9)*(Fl - Fr) )

		return mu, np.array([dx[0], dx[1], dTheta, dTb, dE ])


	def step( self, c_step, h, t ):
		mu, dr = self.dmap( self.state[:, c_step], t )
		self.state[:, c_step + 1] = self.state[:, c_step] + h*dr
		self.state[4] = np.heaviside(self.state[4], 0.5)*self.state[4]
		# moving/ acting
		dTheta = dr[2]

		self.F = 0
		self.theta = self.state[2, c_step + 1]
		self.x = self.state[0, c_step + 1]
		self.y = self.state[1, c_step + 1]
		self.updateSensorPositions( h, mu, self.x, self.y, self.state[2, c_step], self.theta )
		self.F = self.enviroment.getFood( self.x, self.y )

	def draw( self ):
		c = plt.Circle( (self.x, self.y), self.radius, color = 'k' )
		p1 = self.sensors['T_left']
		p2 = self.sensors['T_right']
		cs1 = plt.Circle( (p1[0], p1[1]), 1, color = [0.5,0.5,0.5] )
		cs2 = plt.Circle( (p2[0], p2[1]), 1, color = [0.5,0.5,0.5] )
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
		for i in range(len(self.food_sources)):
			x0,y0 = self.food_sources[i]

			if( np.linalg.norm(np.array([x-x0, y-y0])) < 0.1 ):
				return 1.0
			
		return 0.0

	def draw( self ):
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
		fig, self.ax = plt.subplots(figsize = (10,5) )
		self.ax.set_position([0.07, 0.1, 0.4, 0.7])
		self.ax_temp = plt.axes([0.52, 0.7, 0.46, 0.2])	
		self.ax_energy = plt.axes([0.52, 0.42, 0.46, 0.2])	
		self.observed = None

	def addAgent( self, a, observe = 1.0 ):
		a.enviroment = self.enviroment
		self.agents.append( a )
		self.observed = len(self.agents)-1

	def addFoodSource( self, x, y ):
		self.enviroment.addFoodSource( x, y )

	def draw( self, c_step, tf ):
		# plt.subplots_adjust(left=0.1, bottom=0.25, right = 0.9, top = 0.95)

		plt.ioff()
		plt.cla()
		plt.axis([0, self.enviroment.w, 0, self.enviroment.h])
		#plt.title('Motivational conflict motivation')
		plt.xlabel('x')
		plt.ylabel('y')
		# Setting figure properties
		plt.ion()
		self.enviroment.draw()

		for a in self.agents:
			a.draw()

		a = self.agents[self.observed]	
		plt.sca(self.ax_temp)	
		plt.cla()
		self.ax_temp.plot(self.time[:c_step], a.state[3,:c_step])
		self.ax_temp.axis([0, tf, 0, 40])	
		self.ax_temp.set_title('Tb')	
		plt.sca(self.ax_energy)
		plt.cla()
		self.ax_energy.plot(self.time[:c_step], a.state[4,:c_step])
		self.ax_energy.axis([0, tf, 0, 1.1])
		self.ax_energy.set_title('E')
		plt.sca(self.ax)

		#ax.margins(x = 0)
		plt.pause(0.01)


	def run( self, tf ):
		c_step = 0
		t = 0.0
		m = int(tf/self.h)
		self.time = np.zeros((m, ))

		for a in self.agents:
			a.init( tf, self.h )

		while( c_step < m-1 ):
			self.time[c_step] = t

			for a in self.agents:
				a.step( c_step, self.h, t )

			self.draw( c_step, tf )
			c_step += 1
			t += self.h

		plt.ioff()
		plt.show()

if __name__ == '__main__':
	s = Simulation()
	a = Agent( x = 85.0, y = 50.0, theta = np.pi, Tb = 37.0 )
	s.addAgent( a )
	s.addFoodSource( 20, 20  )
	s.run( 10.0 )