#from abc import ABC

class Sensor(object):
	def __init__(self, ego_angle, environment):
		# Initializes the sensor.
		# ego_angle: Angular position of the sensor in the body
		# environment: An instance of the enviroment to get the redings from
		self.pos = [0,0]
		self.ego_angle = ego_angle
		self.environment = environment

	def setPosition( self, x, y ):
		self.pos = [x,y]

	def getReading( self ):
		pass

class TemperatureSensor( Sensor ):
	def __init__( self, ego_angle, environment ):
		super(TemperatureSensor, self).__init__(ego_angle, environment)

	def getReading( self ):
		return self.environment.getTemperature( self.pos[0], self.pos[1] )

class ChemicalSensor( Sensor ):
	def __init__( self, ego_angle, environment ):
		super(ChemicalSensor, self).__init__(ego_angle, environment)

	def getReading( self ):
		return self.environment.getFoodSignal( self.pos[0], self.pos[1] )

