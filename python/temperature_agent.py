import numpy as np 
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Enviroment: Contains the main behaviour of the agent
# --------------------------------------------------------------------
class Agent:

	def __init__(self, x, y, theta, Tb, Tp = 37.0, radius = 1.0, k1 = 0.99, G = 0.1 ):
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

	def getTransformation( self, x, y, theta ):
		return np.matrix([[np.cos(theta), -np.sin(theta), x], 
						[np.sin(theta), np.cos(theta), y], 
						[0, 0, 1]])

	def updateSensorPositions( self, x, y, theta ):
		M = self.getTransformation( x, y, theta )

		for k in self.sensors:
			self.sensors[k] = np.dot(M, self.sensors[k])

	def initSensors(self, x, y, theta):
		pl = [self.radius, 0, 0]
		pr = np.array([-self.radius, 0, 0])

		M = self.getTransformation( x, y, theta )

		self.sensors = { 'T_left': np.dot(M,pl).T, 
					'T_right': np.dot(M,pr).T, 
					'F_left': np.dot(M,pl).T,
					'F_right': np.dot(M,pr).T }

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
		# State variables
		theta = u[2]
		Tb = u[3]
		E = u[4]
		# Drives
		dHeat = np.abs(Tb - self.Tp)
		dFood = np.heaviside(1 - E, 0.0)*(1 - E)
		mu = np.linalg.norm(np.array([dHeat, dFood]))

		dx = mu*np.array([np.cos(theta), np.sin(theta)])
		k2 = 1.0
		Tc = Tb # No contact
		alpha = 0.1
		dTb = self.G - self.k1*(Tb - Ta)*self.A - k2*(1 - self.A)*(Tb - Tc)
		dE = -alpha*self.G + self.F
		# nonlinearity
		sigma = 0.1
		f = lambda x : 1.0/(1 + np.exp(-sigma*x))

		if( dHeat > dFood ):
			dTheta = f( (Tb - self.Tp)*(Tl - Tr) )
		else:
			dTheta = f( E )

		return np.array([dx[0], dx[1], dTheta, dTb, dE ])


	def step( self, c_step, h, t ):
		dr = self.dmap( self.state[:, c_step], t )
		self.state[:, c_step + 1] = self.state[:, c_step] + h*dr
		# moving/ acting
		self.F = 0
		self.theta = self.state[2, c_step + 1]
		self.x = self.state[0, c_step + 1]
		self.y = self.state[1, c_step + 1]
		self.updateSensorPositions( self.x, self.y, self.theta )
		self.F = self.enviroment.getFood( self.x, self.y )

	def draw( self ):
		c = plt.Circle( (self.x, self.y), self.radius, color = 'k' )
		fig = plt.gcf()
		ax = fig.gca()
		ax.add_artist(c)
		

# --------------------------------------------------------------------
# Enviroment: Manages enviroment drawing and signals
# --------------------------------------------------------------------
class Enviroment:
	def __init__( self ):
		self.w = 100
		self.h = 100
		self.food_sources = []
		sigma = 0.1
		self.g = lambda x,y,x0,y0: np.exp(-(( x - x0)**2 + (y - y0)**2)/(2*sigma**2) )
		self.Tmin = 15.0
		self.Tmax = 45.0
		self.T = lambda x,y : ((self.Tmax - self.Tmin)/self.w)*x + self.Tmin
		

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
		fig = plt.gcf()
		cax = fig.gca()
		axcolor = 'lightgoldenrodyellow'
		ax_gradient = plt.axes([0.1, 0.8, 0.8, 0.09], facecolor=axcolor)	
		ax_gradient.tick_params(bottom = False, left = False, top = False, labelbottom = False, labelleft = False)
		ax_gradient.axis([0, self.w, self.Tmin, self.Tmax])
		xx = [0, self.w, self.w]
		yy = [self.Tmin, self.Tmin, self.Tmax]
		ax_gradient.fill( xx, yy, color=[0.5,0.5,0.7] )


		plt.sca(cax)

		for i in range(len(self.food_sources)):
			x0,y0 = self.food_sources[i]

			x = np.arange(0, self.w, delta)
			y = np.arange(0, self.h, delta)

			X, Y = np.meshgrid(x, y)
			Z = np.exp(-(X - x0*np.ones(X.shape))**2 - (Y - y0*np.ones(Y.shape))**2)
			# Z = self.g(X,x0*np.ones(X.shape),Y, y0*np.ones(Y.shape))

			nr, nc = Z.shape

			# put NaNs in one corner:
			Z[-nr // self.w:, -nc // self.h:] = np.nan
			# contourf will convert these to masked


			Z = np.ma.array(Z)
			# mask another corner:
			Z[:nr // self.w, :nc // self.h] = np.ma.masked

			# mask a circle in the middle:
			interior = np.sqrt(X**2 + Y**2) < 0.5
			Z[interior] = np.ma.masked

			CS = plt.contourf(X, Y, Z, 10, cmap=plt.cm.hot)



# --------------------------------------------------------------------
# Simlation: Manages the simulation
# --------------------------------------------------------------------
class Simulation:
	def __init__(self):
		self.h = 0.01
		self.agents = []
		self.enviroment = Enviroment()
		fig, ax = plt.subplots(figsize = (6,6) )
		ax.set_position([0.1, 0.1, 0.8, 0.7])

	def addAgent( self, a ):
		a.enviroment = self.enviroment
		self.agents.append( a )

	def addFoodSource( self, x, y ):
		self.enviroment.addFoodSource( x, y )

	def draw( self ):
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

		#ax.margins(x = 0)
		plt.pause(0.01)


	def run( self, tf ):
		c_step = 0
		t = 0.0

		for a in self.agents:
			a.init( tf, self.h )

		m = tf//self.h

		while( c_step < m - 1 ):
			for a in self.agents:
				a.step( c_step, self.h, t )

			self.draw()
			c_step += 1

if __name__ == '__main__':
	s = Simulation()
	a = Agent( x = 50.0, y = 50.0, theta = 0.0, Tb = 37.0 )
	s.addAgent( a )
	s.addFoodSource( 1, 90  )
	s.run( 10.0 )