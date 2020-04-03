import numpy as np

class Body:
	def __init__( self, x0 = 0.0, y0 = 0.0, radius = 2.0, theta0 = 0, dtheta0 = 0):
		v0 = 0.0

		self.theta0 = theta0
		self.dtheta0 = dtheta0
		self.v0 = v0
		self.x0 = np.array([x0, y0])
		self.theta = theta0
		self.dtheta = dtheta0
		self.radius = radius
		self.v = v0
		self.dx = v0
		self.x = np.array([x0, y0])
		self.gamma = 100.0
		self.mu = 0.5
		self.m = 1.0
		self.kappa = 0.1
		self.A = 2*np.pi*radius

	def reset( self ):
		self.theta = self.theta0
		self.dtheta = self.dtheta0
		self.v = self.v0
		self.x = self.x0

	def step( self, c_step, h, drive_wheels ):
	# drive_wheels expected to be a two dimensional vector v \in [0,1]x[0,1]

		d = self.radius
		Ic = self.m*self.kappa**2.0
		F1 = self.gamma*drive_wheels[0,0]
		F2 = self.gamma*drive_wheels[0,1]

		Fmu = self.mu*np.linalg.norm(self.v) if \
			  (np.linalg.norm(self.v) > 0.0)  else 0.0
		#and (F1 < 0.1 and F2 < 0.1)

		print "Fmu: ", Fmu
		Fmu_theta = 10*self.dtheta if \
			  (np.linalg.norm(self.dtheta) > 0.0) else 0.0

		F_left = F1 - Fmu
		F_right = F2 - Fmu
		dv = (F_left + F_right)/self.m
		ddtheta = (d*F_right - d*F_left)/Ic - Fmu_theta

		# if all(ddtheta) > 0.0:
		# 	r = -dv/ddtheta
		# else:
		# 	r = x

		self.v += h*dv
		self.dtheta += h*ddtheta
		self.theta += h*self.dtheta
		self.dx = np.array([self.v*np.cos(self.theta), 
			  				self.v*np.sin(self.theta)])

		self.x += h*self.dx
		return self.x, self.dx # Returns current position

	def getPosition( self ):
	# Returns the postion as a tuple
		return self.x[0], self.x[1]

	def getOrientation( self ):
		return self.theta