import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------
# Enviroment: Manages enviroment drawing and signals
# --------------------------------------------------------------------
class Environment:
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
		xx = [0 + offset[0], self.w + offset[0], self.w + offset[0]]
		yy = [self.Tmin + offset[0], self.Tmin + offset[0], self.Tmax + offset[0]]
		self.ax_gradient.fill( xx, yy, color=[0.8,0.5,0.5] )

		plt.sca(cax)

		for i in range(len(self.food_sources)):
			plt.sca( self.ax_gradient )
			#self.ax_gradient.axis([0, self.w, self.Tmin, self.Tmax])
			plt.yticks([self.Tmin, self.Tmax], [self.Tmin - offset[0], self.Tmax - offset[0]])

			plt.sca(cax)
			x0,y0 = self.food_sources[i]
			x0 += offset[0]
			y0 += offset[1]

			x = np.arange(0 + offset[0], self.w + offset[0], delta)
			y = np.arange(0 + offset[1], self.h + offset[1], delta)

			X, Y = np.meshgrid(x, y)
			#Z = np.exp(-(X - x0*np.ones(X.shape))**2 - (Y - y0*np.ones(Y.shape))**2)
			Z = self.g(X,Y, x0*np.ones(X.shape), y0*np.ones(Y.shape))		
			CS = plt.contour(X, Y, Z, origin = 'lower' )

			c = plt.Circle( (x0, y0), 1.0, color = 'g' )
			fig = plt.gcf()
			ax = fig.gca()
			ax.add_artist(c)
