import numpy as np 
import matplotlib.pyplot as plt
import sys

def plot_general_potential():
    c = 1.0
    d = 0.0
    e = 0.0
    U = lambda x: (1.0/4.0)*((x**2)*(1-x)**2*(2-x)**2 + (1-x)**2*c + x**2*d + (2-x)**2*e)
    x = np.linspace(-0.2,2.2,100)

    plt.plot( x, [U(s) for s in x] )
    plt.show()


def analysis_minima2( a = 0.1, b = 0.5 ):
    fig, ax = plt.subplots(1, 2)
    r1 = 0.0
    r2 = 1.0
    U = lambda x: ((r1 - x)**2*(r2 - x)**2 + a*(r1 - x)**2 + b*(r2-x)**2) 
    Q = lambda x: -(r1 - x)*(r2 - x)*((r1 + r2)/2.0 - x)
    L = lambda x: -(a + b)*x/2.0 + (a*r1 + b*r2)/2.0
    x = np.linspace(-0.2,1.2,100)
    ax[0].plot( x, U(x), linewidth=3.0 )
    ax[0].grid()
    ax[0].set_title('Motivational potential')
    ax[0].set_xlabel('Motivational space')

    ax[1].plot( x, Q(x), linewidth=3.0 )
    ax[1].plot( x, L(x), linewidth=3.0 )
    ax[1].set_title('Well relation')
    ax[1].set_xlabel('Motivational space')

    ax[1].grid()
    plt.show()

def analysis_minima3( a = 0.1, b = 0.5, c = 0.1 ):
    fig, ax = plt.subplots(1, 2)

    r1 = -1.0
    r2 = 0.0
    r3 = 1.0

    U = lambda x: ((r1 - x)**2*(r2-x)**2*(r3-x)**2 + a*(r1 - x)**2 + b*(r2-x)**2 + c*(r3-x)**2) 
    Q = lambda x: (r1 - x)*(r2 - x)*(r3 - x)*(-(r2 - x)*(r3 - x) - (r1 - x)*(r2 + r3 - 2.0*x))
    L = lambda x: -(a + b + c)*x + r1*a + r2*b + r3*c
    x = np.linspace(r1-0.1,r3+0.1,100)
    ax[0].plot( x, U(x) )
    ax[0].grid()

    ax[1].plot( x, Q(x) )
    ax[1].plot( x, L(x) )

    ax[1].grid()
    plt.show()

def plot_inhibition_potential():

    p = np.linspace(0.0,1.0, 20)
    b = 1.0
    x = np.linspace(-0.5,1.5,100)
    plt.ion()

    for i in range(len(p)):
        a = 1.0-p[i]
        U = lambda x: (1.0/4.0)*((x**2)*((1-x)**2) + b*(1-x)**2 + a*x**2)    

        plt.plot( x, [U(s) for s in x] )
        plt.show()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

def plot_disinhibition_potential():
    pass


if __name__ == '__main__':
    v = sys.argv

    if len(v) == 3:
        a = float(v[1])
        b = float(v[2])
        analysis_minima2(a, b)    
    elif len(v) == 4:
        a = float(v[1])
        b = float(v[2])
        c = float(v[3])
        analysis_minima3(a, b, c)    
    else:
        print "Wrong amount of arguments"

    #plot_inhibition_potential()
    

