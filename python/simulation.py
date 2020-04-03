import numpy as np 
import matplotlib.pyplot as plt
# Maps
from brainsv2 import *
from enviroment import *
from sensors import *
from recorder import *
from agent import *
import time


# --------------------------------------------------------------------
# Simlation: Manages the simulation
# --------------------------------------------------------------------
class Simulation:
	def __init__(self):
		self.h = 0.01
		self.agents = []
		self.environment = Environment()

		# self.fig2, self.ax2 = plt.subplots(2,2, figsize = (10,5))
		self.fig1, self.ax = plt.subplots(figsize = (10,5) )
		self.ax.set_position([0.07, 0.1, 0.4, 0.7])
		self.ax_temp = plt.axes([0.52, 0.7, 0.46, 0.2])	
		self.ax_energy = plt.axes([0.52, 0.42, 0.46, 0.2])	
		self.ax_potential = plt.axes([0.52, 0.10, 0.46, 0.2])
		self.observed = None
		self.offset = np.array([0,0])

	def addAgent( self, a, observe = 1.0 ):
		a.envirnoment = self.environment
		self.agents.append( a )
		self.observed = len(self.agents)-1

	def addFoodSource( self, x, y ):
		self.environment.addFoodSource( x, y )

	def draw( self, c_step, tf ):
		# plt.subplots_adjust(left=0.1, bottom=0.25, right = 0.9, top = 0.95)
		#plt.ioff()

		a = self.agents[0]

		plt.sca(self.ax)
		plt.cla()
		plt.axis([0, self.environment.w, 0, self.environment.h])
		#plt.title('Motivational conflict motivation')
		plt.xlabel('x')
		plt.ylabel('y')
		# Setting figure properties
		plt.ion()		

		for a in self.agents:
			(x,y) = a.body.getPosition()

			if x + self.offset[0] > self.environment.w :
				self.offset[0] = self.environment.w + self.offset[0] - x - 10.0
			if y + self.offset[1] > self.environment.h:
				self.offset[1] = self.environment.h + self.offset[1] - y - 10.0
			if x + self.offset[0] < 0:
				self.offset[0] = -x + self.offset[0] + 10.0
			if y + self.offset[1] < 0:
				self.offset[1] = -y + self.offset[1] + 10.0


		self.environment.draw( self.offset )

		for a in self.agents:
			a.draw( c_step, self.offset )

		a = self.agents[self.observed]	
		a.recorder.plotStateVariable( self.ax_temp, 'Tb', self.time, c_step, axis = [0, tf, 0, 40.0] )
		a.recorder.plotStateVariable( self.ax_energy, 'E', self.time, c_step, axis = [0, tf, 0, 1.0] )
		

		# if a.U is not None:
		# 	x = np.linspace(-1.0, 2.0, 100)
		# 	self.ax_potential.plot( x, a.U(x))

		# self.ax_potential.set_title('Potential')
		# plt.sca(self.ax)

		#ax.margins(x = 0)
		plt.pause(0.001)


	def run( self, tf ):
		c_step = 0
		t = 0.0
		m = int(tf/self.h)
		self.time = np.zeros((m, ))

		for a in self.agents:
			a.reset( tf, self.h )

		while( c_step < m-1 ):
			self.time[c_step] = t

			for a in self.agents:
				a.step( c_step, self.h, t )

			self.draw( c_step, tf )
			c_step += 1
			t += self.h

		plt.ioff()
		plt.show()