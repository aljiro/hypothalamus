from simulation import *
from agent import *
import sys

if __name__ == '__main__':

	s = Simulation()
	
	if len(sys.argv) == 1 or sys.argv[1] == 'simple':
		a = AgentBuilder.buildSimpleTemperatureAgent( s.environment, 20.0, 20.0 )
	else:
		if sys.argv[1] == 'motivational':
			a = AgentBuilder.buildMotivationalAgent( s.environment, 20.0, 20.0 )
		else:
			error('Unrecognized argument')

	
	s.addAgent( a )
	s.addFoodSource( 40, 40  )
	s.run( 20.0 )
