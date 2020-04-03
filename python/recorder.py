import numpy as np
import matplotlib.pyplot as plt

class Recorder:
	def __init__( self, state_labels ):
		self.data = {}
		self.state_labels = state_labels
		

	def reset( self, m, state, variables ):
		self.trace = np.zeros( (2, m) )		
		self.state = np.zeros( (len(self.state_labels), m) )

		for v in variables:
			self.data[v] = np.zeros(m)

	def recordState( self, state, step ):
		self.state[:,step] = state

	def recordPosition( self, x, y, step ):
		self.trace[:,step] = [x,y]

	def recordVariable( self, name, value, step ):
		if name not in self.data:
			raise Exception('Recorder exception', 'The variable has not been set up to be recorded')

		self.data[name][step] = value 

	def plotStateVariable( self, ax, name, time, step, axis = None ):
		if len(self.state_labels) == 0:
			return 
			
		if name not in self.state_labels:
			raise Exception('Recorder error', 'Name not a state variable')

		idx = self.state_labels.index(name)
		plt.sca(ax)	
		plt.cla()
		ax.plot(time[:step], self.state[idx,:step])

		if axis is not None:
			ax.axis(axis)	
			
		ax.set_title(name)

	def plotTrace( self, ax, step, offset ):
		ax.plot( self.trace[0,:step] + offset[0], self.trace[1,:step] + offset[1], 'k--')
