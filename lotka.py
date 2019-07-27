from numpy import *
import pylab as p
from scipy import integrate
# Definition of parameters 
k = [0.00231, 0.00731, 0.0231, 0.0731, 0.231]
a = -0.6931
b = 0.6931 
c = k[1]
d = 0.5



#!python
def d2X_dt2(X, t=0):
    """ Return the Jacobian matrix evaluated in X. """
    return array([[a -b*X[1],   -b*X[0]     ],
                  [b*d*X[1] ,   -c +b*d*X[0]] ])
def dX_dt(X, t=0):
    return array([ a*X[0] - b*X[0]*X[1] , -c*X[1] + d*b*X[0]*X[1] ])

#


t = linspace(0, 15,  1000) # time
X0 = array([1, 0])

f2 = p.figure()
values  = linspace(0,2, 10)                          # position of X0 between X_f0 and X_f1
vcolors = p.cm.autumn_r(linspace(0.3, 1., len(values)))  # colors for each trajectory
# in plot trajectories

#!python
ymax = p.ylim(ymin=0)[1]                        # get axis limits
xmax = p.xlim(xmin=0)[1] 
nb_points   = 20                      

x = linspace(0, xmax, nb_points)
y = linspace(0, ymax, nb_points)

X1 , Y1  = meshgrid(x, y)                       
DX1, DY1 = dX_dt([X1, Y1])                      
M = (hypot(DX1, DY1))                           
M[ M == 0] = 1.                                 
DX1 /= M # Normalize each arrows
DY1 /= M     
X_f0 = array([     0. ,  0.])
X_f1 = array([ c/(d*b), a/b])
all(dX_dt(X_f0) == zeros(2) ) and all(dX_dt(X_f1) == zeros(2)) # => True

def IF(X):
    u, v = X
    return u**(c/a) * v * exp( -(b/a)*(d*u+v) )

for v, col in zip(values, vcolors): 
    X0 = v * X_f1 # starting point
    X = integrate.odeint( dX_dt, X0, t) 
    p.plot( X[:,0], X[:,1], lw=3.5*v, color=col, label='X0=(%.f, %.f)' % ( X0[0], X0[1]) )

X2 , Y2  = meshgrid(x, y)
Z2 = IF([X2, Y2])  
# Drow direction fields, using matplotlib 's quiver function
# color: growth speed
CS = p.contourf(X2, Y2, Z2, cmap=p.cm.Purples_r, alpha=0.5)
CS2 = p.contour(X2, Y2, Z2, colors='black', linewidths=2. )
p.clabel(CS2, inline=1, fontsize=16, fmt='%.f')
p.title('Trajectories and direction fields')
Q = p.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)
# p.xlabel('Number of rabbits')
# p.ylabel('Number of foxes')
p.legend()
p.grid()
p.xlim(0, xmax)
p.ylim(0, ymax)
p.show()
f2.savefig('rabbits_and_foxes_2.png')
